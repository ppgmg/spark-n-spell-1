# contextSPARK1.py - SPARK implementation, naive parallelization

import re
import math
from scipy.stats import poisson
import time
import sys, getopt
import os

# Initialize Spark
from pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel('ERROR')

######################
#
# Submission by Gioia Dominedo (Harvard ID: 40966234) for
# CS 205 - Computing Foundations for Computational Science
# 
# This is part of a joint project with Kendrick Lo that includes a
# separate component for word-level checking. This script includes 
# one of three SPARK implementations for context-level spell-checking
# adapted from third party algorithms (Symspell and Viterbi algorithms). 
#
# The following were also used as references:
# Peter Norvig, How to Write a Spelling Corrector
#   (http://norvig.com/spell-correct.html)
# Peter Norvig, Natural Language Corpus Data: Beautiful Data
#   (http://norvig.com/ngrams/ch14.pdf)
#
######################

######################
#
# SUMMARY OF CONTEXT-LEVEL CORRECTION LOGIC - VITERBI ALGORITHM
#
# v 1.0 last revised 6 Dec 2015
#
# Each sentence is modeled as a hidden Markov model. Prior
# probabilities (for first words in the sentences) and transition
# probabilities (for all subsequent words) are calculated when
# generating the main dictionary, using the same corpus. Emission
# probabilities are generated on the fly by parameterizing a Poisson 
# distribution with the edit distance between words and suggested
# corrections.
#
# The state space of possible corrections for each word is generated
# using logic based on the Symspell spell-checker (see below for more
# detail on Symspell). Valid suggestions must: (a) be 'real' words;
# (b) appear at least 100 times in the corpus used to generate the
# dictionary; (c) be one of the top 10 suggestions, based on frequency
# and edit distance. This simplification ensures that the state space
# remains manageable.
#
# All probabilities are stored in log-space to avoid underflow. Pre-
# defined minimum values are used for words that are not present in
# the dictionary and/or probability tables.
#
# The pre-processing steps are the same across all three SPARK
# implementations.
#
# More detail on the specific implementation is included below.
#
######################

######################
#
# SPARK IMPLEMENTATION DETAILS - NAIVE PARALLELIZATION
#
# This is the first attempt at parallelizing the Viterbi algorithm.
#
# For this implementation we create an RDD where each element
# corresponds to a sentence from the document to be checked, and
# then use a map operation to call the Viterbi function for each
# sentence.
#
# This approach takes advantage of parallelization by splitting the
# sentences among the workers (i.e. increasing the number of workers
# will improve the runtime), but does not parallelize the algorithm
# itself.
#
# Note: all the functions related to the Viterbi algorithm are the
# same as those used in the serial implementation.
#
######################

######################
#
# PRE-PROCESSING STEPS
#
# The pre-processing steps have been adapted from the dictionary
# creation of the word-level spellchecker, which in turn was based on
# SymSpell, a Symmetric Delete spelling correction algorithm
# developed by Wolf Garbe and originally written in C#. More detail
# on SymSpell is included in the word-level spellcheck documentation.
#
# The main modifications to the word-level spellchecker pre-
# processing stages are to create the additional outputs that are
# required for the context-level checking, and to eliminate redundant
# outputs that are not necessary.
#
# The outputs of the pre-processing stage are:
#
# - dictionary: A dictionary that combines both words present in the
# corpus and other words that are within a given 'delete distance'. 
# The format of the dictionary is:
# {word: ([list of words within the given 'delete distance'], 
# word count in corpus)}
#
# - start_prob: A dictionary with key, value pairs that correspond to
# (word, probability of the word being the first word in a sentence)
#
# - transition_prob: A dictionary of dictionaries that stores the
# probability of a given word following another. The format of the
# dictionary is:
# {previous word: {word1 : P(word1|prevous word), word2 : 
# P(word2|prevous word), ...}}
#
# - default_start_prob: A benchmark probability of a word being at
# the start of a sentence, set to 1 / # of words at the beginning of
# sentences. This ensures that all previously unseen words at the
# beginning of sentences are not corrected unnecessarily.
#
# - default_transition_prob: A benchmark probability of a word being
# seen, given the previous word in the sentence, also set to 1 / # of
# transitions in corpus. This ensures that all previously unseen
# transitions are not corrected unnecessarily.
#
######################

def get_deletes_list(w, max_edit_distance):
    '''
    Given a word, derive strings with up to max_edit_distance
    characters deleted. 

    The list is generally of the same magnitude as the number of
    characters in a word, so it does not make sense to parallelize
    this function. Instead, we use Python to create the list.
    '''
    deletes = []
    queue = [w]
    for d in range(max_edit_distance):
        temp_queue = []
        for word in queue:
            if len(word)>1:
                for c in range(len(word)):  # character index
                    word_minus_c = word[:c] + word[c+1:]
                    if word_minus_c not in deletes:
                        deletes.append(word_minus_c)
                    if word_minus_c not in temp_queue:
                        temp_queue.append(word_minus_c)
        queue = temp_queue
        
    return deletes

def get_transitions(sentence):
    '''
    Helper function: converts a sentence into all two-word pairs.
    Output format is a list of tuples.
    e.g. 'This is a test' >> ('this', 'is'), ('is', 'a'), ('a', 'test')
    ''' 
    if len(sentence)<2:
        return None
    else:
        return [((sentence[i], sentence[i+1]), 1) 
                for i in range(len(sentence)-1)]
    
def map_transition_prob(vals):
    '''
    Helper function: calculates conditional probabilities for all word
    pairs, i.e. P(word|previous word)
    '''
    total = float(sum(vals.values()))
    return {k: math.log(v/total) for k, v in vals.items()}

def parallel_create_dictionary(fname, max_edit_distance=3, 
                                num_partitions=6):
    '''
    Load a text file and use it to create a dictionary and
    to calculate start probabilities and transition probabilities. 
    '''
    
    # Note: this function makes use of multiple accumulators to keep
    # track of the words that are being processed. An alternative 
    # implementation that wraps accumulators in helper functions was
    # also tested, but did not yield any noticeable improvements.

    ############
    #
    # load file & initial processing
    #
    ############
    
    # http://stackoverflow.com/questions/22520932/python-remove-all-non-alphabet-chars-from-string
    regex = re.compile('[^a-z ]')

    # load file contents and convert into one long sequence of words
    # RDD format: 'line 1', 'line 2', 'line 3', ...
    # cache because this RDD is used in multiple operations 
    make_all_lower = sc.textFile(fname) \
            .map(lambda line: line.lower()) \
            .filter(lambda x: x!='').cache()
    
    # split into individual sentences and remove other punctuation
    # RDD format: [words of sentence 1], [words of sentence 2], ...
    # cache because this RDD is used in multiple operations 
    split_sentence = make_all_lower.flatMap(lambda 
        line: line.replace('?','.').replace('!','.').split('.')) \
             .map(lambda sentence: regex.sub(' ', sentence)) \
             .map(lambda sentence: sentence.split()) \
             .filter(lambda x: x!=[]).cache()
    
    ############
    #
    # generate start probabilities
    #
    ############
    
    # extract all words that are at the beginning of sentences
    # RDD format: 'word1', 'word2', 'word3', ...
    start_words = split_sentence.map(lambda sentence: sentence[0] 
        if len(sentence)>0 else None) \
            .filter(lambda word: word!=None)
    
    # add a count to each word
    # RDD format: ('word1', 1), ('word2', 1), ('word3', 1), ...
    # note: partition here because we are using words as keys for
    # the first time - yields a small but consistent improvement in
    # runtime (~2-3 sec for big.txt)
    # cache because this RDD is used in multiple operations
    count_start_words_once = start_words.map(lambda word: (word, 1)) \
            .partitionBy(num_partitions).cache()

    # use accumulator to count the number of start words processed
    accum_total_start_words = sc.accumulator(0)
    count_start_words_once.foreach(lambda x: accum_total_start_words.add(1))
    total_start_words = float(accum_total_start_words.value)
    
    # reduce into count of unique words at the start of sentences
    # RDD format: ('word1', frequency), ('word2', frequency), ...
    unique_start_words = count_start_words_once.reduceByKey(lambda a, b: a + b)
    
    # convert counts to log-probabilities
    # RDD format: ('word1', log-prob of word1), 
    #             ('word2', log-prob of word2), ...
    start_prob_calc = unique_start_words.mapValues(lambda v: 
        math.log(v/total_start_words))
    
    # get default start probabilities (for words not in corpus)
    default_start_prob = math.log(1/total_start_words)
    
    # store start probabilities as a dictionary (i.e. a lookup table)
    # note: given the spell-checking algorithm, this cannot be maintained
    # as an RDD as it is not possible to map within a map
    start_prob = start_prob_calc.collectAsMap()
    
    ############
    #
    # generate transition probabilities
    #
    ############
    
    # note: various partitioning strategies were attempted for this
    # portion of the function, but they failed to yield significant
    # improvements in performance.

    # focus on continuous word pairs within the sentence
    # e.g. "this is a test" -> "this is", "is a", "a test"
    # note: as the relevant probability is P(word|previous word)
    # the tuples are ordered as (previous word, word)

    # extract all word pairs within a sentence and add a count
    # RDD format: (('word1', 'word2'), 1), (('word2', 'word3'), 1), ...
    # cache because this RDD is used in multiple operations 
    other_words = split_sentence.map(lambda sentence: 
        get_transitions(sentence)) \
            .filter(lambda x: x!=None) \
            .flatMap(lambda x: x).cache()

    # use accumulator to count the number of transitions (word pairs)
    accum_total_other_words = sc.accumulator(0)
    other_words.foreach(lambda x: accum_total_other_words.add(1))
    total_other_words = float(accum_total_other_words.value)
    
    # reduce into count of unique word pairs
    # RDD format: (('word1', 'word2'), frequency), 
    #             (('word2', 'word3'), frequency), ...
    unique_other_words = other_words.reduceByKey(lambda a, b: a + b)
    
    # aggregate by (and change key to) previous word
    # RDD format: ('previous word', {'word1': word pair count, 
    #                                'word2': word pair count}}), ...
    other_words_collapsed = unique_other_words.map(lambda x: 
        (x[0][0], (x[0][1], x[1]))) \
            .groupByKey().mapValues(dict)

    # note: the above line of code is the slowest in the function
    # (8.6 MB shuffle read and 4.5 MB shuffle write for big.txt)
    # An alternative approach that aggregates lists with reduceByKey was
    # attempted, but did not yield noticeable improvements in runtime.
    
    # convert counts to log-probabilities
    # RDD format: ('previous word', {'word1': log-prob of pair, 
    #                                 word2: log-prob of pair}}), ...
    transition_prob_calc = other_words_collapsed.mapValues(lambda v: 
        map_transition_prob(v))

    # get default transition probabilities (for word pairs not in corpus)
    default_transition_prob = math.log(1/total_other_words)
    
    # store transition probabilities as a dictionary (i.e. a lookup table)
    # note: given the spell-checking algorithm, this cannot be maintained
    # as an RDD as it is not possible to map within a map
    transition_prob = transition_prob_calc.collectAsMap()
    
    ############
    #
    # generate dictionary
    #
    ############

    # note: this approach is slightly different from the original SymSpell
    # algorithm, but is more appropriate for a SPARK implementation
    
    # split into individual words (all)
    # RDD format: 'word1', 'word2', 'word3', ...
    # cache because this RDD is used in multiple operations 
    all_words = make_all_lower.map(lambda line: regex.sub(' ', line)) \
            .flatMap(lambda line: line.split()).cache()

    # use accumulator to count the number of words processed
    accum_words_processed = sc.accumulator(0)
    all_words.foreach(lambda x: accum_words_processed.add(1))

    # add a count to each word
    # RDD format: ('word1', 1), ('word2', 1), ('word3', 1), ...
    count_once = all_words.map(lambda word: (word, 1))

    # reduce into counts of unique words - this is the core corpus dictionary
    # (i.e. only words appearing in the file, without 'deletes'))
    # RDD format: ('word1', frequency), ('word2', frequency), ...
    # cache because this RDD is used in multiple operations 
    # note: imposing partitioning at this step yields a small 
    # improvement in runtime (~1 sec for big.txt) by equally
    # balancing elements among workers for subsequent operations
    unique_words_with_count = count_once.reduceByKey(lambda a, b: a + b, 
        numPartitions = num_partitions).cache()
    
    # use accumulator to count the number of unique words
    accum_unique_words = sc.accumulator(0)
    unique_words_with_count.foreach(lambda x: accum_unique_words.add(1))

    # generate list of "deletes" for each word in the corpus
    # RDD format: (word1, [deletes for word1]), (word2, [deletes for word2]), ...
    generate_deletes = unique_words_with_count.map(lambda (parent, count): 
        (parent, get_deletes_list(parent, max_edit_distance)))
    
    # split into all key-value pairs
    # RDD format: (word1, delete1), (word1, delete2), ...
    expand_deletes = generate_deletes.flatMapValues(lambda x: x)
    
    # swap word order and add a zero count (because "deletes" were not
    # present in the dictionary)
    swap = expand_deletes.map(lambda (orig, delete): (delete, ([orig], 0)))
    
    # create a placeholder for each real word
    # RDD format: ('word1', ([], frequency)), ('word2', ([], frequency)), ...
    corpus = unique_words_with_count.mapValues(lambda count: ([], count))

    # combine main dictionary and "deletes" (and eliminate duplicates)
    # RDD format: ('word1', ([deletes for word1], frequency)), 
    #             ('word2', ([deletes for word2], frequency)), ...
    combine = swap.union(corpus)
    
    # store dictionary items and deletes as a dictionary (i.e. a lookup table)
    # note: given the spell-checking algorithm, this cannot be maintained
    # as an RDD as it is not possible to map within a map
    # note: use reduceByKeyLocally to avoid an extra shuffle from reduceByKey
    dictionary = combine.reduceByKeyLocally(lambda a, b: (a[0]+b[0], a[1]+b[1])) 
    
    # output stats
    print 'Total words processed: %i' % accum_words_processed.value
    print 'Total unique words in corpus: %i' % accum_unique_words.value 
    print 'Total items in dictionary (corpus words and deletions): %i' \
        % len(dictionary)
    print '  Edit distance for deletions: %i' % max_edit_distance
    print 'Total unique words at the start of a sentence: %i' \
        % len(start_prob)
    print 'Total unique word transitions: %i' % len(transition_prob)
    
    return dictionary, start_prob, default_start_prob, \
            transition_prob, default_transition_prob

######################
#
# SPELL-CHECKING - VITERBI ALGORITHM
#
# The below functions are used to read in a text file, break it down
# into individual sentences, and then carry out context-based spell-
# checking on each sentence in turn. In cases where the 'suggested'
# word does not match the actual word in the text, both the original
# and the suggested sentences are printed/outputed to file.
#
# Probabilistic model:
#
# Each sentence is modeled as a hidden Markov model, where the
# hidden states are the words that the user intended to type, and
# the emissions are the words that were actually typed.
#
# For each word in a sentence, we can define:
#
# - emission probabilities: P(observed word|intended word)
#
# - prior probabilities (for first words in sentences only):
# P(being the first word in a sentence)
#
# - transition probabilities (for all subsequent words):
# P(intended word|previous intended word)
#
# Prior and transition probabilities were calculated in the pre-
# processing steps above, using the same corpus as the dictionary.
# 
# Emission probabilities are calculated on the fly using a Poisson
# distribution as follows:
# P(observed word|intended word) = PMF of Poisson(k, l), where
# k = edit distance between word typed and word intended, and l=0.01.
# Both the overall approach and the parameter of l=0.01 are based on
# the 2015 lecture notes from AM207 Stochastic Optimization.
# Various parameters for lambda between 0 and 1 were tested, which
# confirmed that 0.01 yields the most accurate word suggestions.
#
# All probabilities are stored in log-space to avoid underflow. Pre-
# defined minimum values (also defined at the pre-processing stage)
# are used for words that are not present in the dictionary and/or
# probability tables.
#
# Algorithm:
#
# The spell-checking itself is carried out using a modified version
# of the Viterbi algorithm, which yields the most likely sequence of
# hidden states, i.e. the most likely sequence of words that form a
# sentence. The main difference to the 'standard' Viterbi algorithm
# is that the state space (i.e. the list of possible corrections) is
# generated (and therefore varies) for each word. This is in contrast
# to the alternative of considering the state space of all possible
# words in the dictionary for every word that is checked, which would
# be intractable for larger dictionaries.
#
# Example:
#
# The algorithm is best illustrated by way of an example.
#
# Suppose that we are checking the sentence 'This is ax test.'
# The emissions for the entire sentence are 'This is ax test.' and
# the hidden states for the entire sentence are 'This is a test.'
#
# As a pre-processing step, we convert everything to lowercase,
# eliminate punctuation, and break the sentence up into a list of
# words: ['this', 'is', 'ax', 'text']
# This list is passed as a parameter to the viterbi function.
#
# The algorithm tackles each word in turn, starting with 'this'.
#
# We first use get_suggestions to obtain a list of all words that
# may have been intended instead of 'this', i.e. all possible hidden
# states (intended words) for the emission (word typed).
#
# get_suggestions returns the 10 most likely corrections:
# - 1 word with an edit distance of 0
#   ['this']
# - 3 words with an edit distance of 1
#   ['his', 'thus', 'thin']
# - 6 words with an edit distance of 2 
#   ['the', 'that', 'is', 'him', 'they', 'their']
# 
# These 10 words represent our state space, i.e. possible words that
# may have been intended, and are referred to below as the list of
# possible corrections. They each have an emission probability equal
# to the PMF of Poisson(edit distance, 0.01).
#
# For each word in the list of possible corrections, we calculate:
# P(word starting a sentence) * P(observed 'this'|intended word)
# This is a simple application of Bayes' rule: by normalizing the
# probabilities we obtain P(intended word|oberved 'this') for
# each of the 10 words.
#
# We store the word-probability pairs for future use, and move on to
# the next word. 
#
# After the first word, all subsequent words are treated as follows.
#
# The second word in our test sentence is 'is'. Once again, we use
# get_suggestions to obtain a list of all words that may have been
# intended. get_suggestions returns the 10 most likely suggestions:
# - 1 word with an edit distance of 0
#   ['is']
# - 9 words with an edit distance of 1
#   ['in', 'it', 'his', 'as', 'i', 's', 'if', 'its', 'us']
# These 10 words represent our state space for the second word.
#
# For each word in the current list of possible corrections, we loop
# through all the words in the previous list of possible corrections,
# and calculate:
#    probability(previous suggested word) 
#    * P(current suggested word|previous suggested word)
#    * P(typing 'is'|meaning to type current suggested word)
# We determine which previous word maximizes this calculation and
# store that 'path' and probability for each current suggested word.
#
# For example, suppose that we are considering the possibility that
# 'is' was indeed intended to be 'is'. We then calculate: 
#    probability(previous suggested word)
#    * P('is'|previous suggested word) * P('is'|'is')
# for all previous suggested words, and discover that the previous
# suggested word 'this' maximizes the above calculation. We therefore
# store 'this is' as the optimal path for the suggested correction
# 'is' and the above (normalized) probability associated with this
# path.
#
# If the sentence had been only 2 words long, then at this point we
# would return the path that maximizes the most probability for the
# most recent step (word).
#
# As it is not, we repeat the previous steps for 'ax' and 'test',
# and then return the path that is associated with the highest
# probability at the last step.
#
######################

def dameraulevenshtein(seq1, seq2):
    '''
    Calculate the Damerau-Levenshtein distance between sequences.

    codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1
    matrix. However, only the current and two previous rows are
    needed at once, so we only store those.

    Same code as word-level checking.
    '''
    
    oneago = None
    thisrow = range(1, len(seq2) + 1) + [0]
    
    for x in xrange(len(seq1)):
        
        twoago, oneago, thisrow = \
            oneago, thisrow, [0] * len(seq2) + [x + 1]
        
        for y in xrange(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)

            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
                
    return thisrow[len(seq2) - 1]

def get_suggestions(string, dictionary, max_edit_distance, 
                    longest_word_length=20, min_count=100, max_sug=10):
    '''
    Return list of suggested corrections for potentially incorrectly
    spelled word.

    Code based on get_suggestions function from word-level checking,
    with the addition of the min_count and max_sug parameters.
    - min_count: minimum number of times a word must have appeared
    in the dictionary corpus to be considered a valid suggestion
    - max_sug: number of suggestions that are returned (ranked by
    frequency of appearance in dictionary corpus and edit distance
    from word being checked)

    These changes were imposed in order to ensure that the problem
    remains tractable when checking very large documents. In practice,
    the "correct" suggestion is almost always amongst the top ten.

    '''
    
    if (len(string) - longest_word_length) > max_edit_distance:
        # to ensure Viterbi can keep running -- use the word itself
        return [(string, 0)]
    
    suggest_dict = {}
    
    queue = [string]
    q_dictionary = {}  # items other than string that we've checked
    
    while len(queue)>0:
        q_item = queue[0]  # pop
        queue = queue[1:]
        
        # process queue item
        if (q_item in dictionary) and (q_item not in suggest_dict):
            if (dictionary[q_item][1]>0):
            # word is in dictionary, and is a word from the corpus,
            # and not already in suggestion list so add to suggestion
            # dictionary, indexed by the word with value (frequency
            # in corpus, edit distance)
            # note: q_items that are not the input string are shorter
            # than input string since only deletes are added (unless
            # manual dictionary corrections are added)
                assert len(string)>=len(q_item)
                suggest_dict[q_item] = \
                    (dictionary[q_item][1], len(string) - len(q_item))
            
            # the suggested corrections for q_item as stored in
            # dictionary (whether or not q_item itself is a valid
            # word or merely a delete) can be valid corrections
            for sc_item in dictionary[q_item][0]:
                if (sc_item not in suggest_dict):
                    
                    # compute edit distance
                    # suggested items should always be longer (unless
                    # manual corrections are added)
                    assert len(sc_item)>len(q_item)
                    # q_items that are not input should be shorter
                    # than original string 
                    # (unless manual corrections added)
                    assert len(q_item)<=len(string)
                    if len(q_item)==len(string):
                        assert q_item==string
                        item_dist = len(sc_item) - len(q_item)

                    # item in suggestions list should not be the same
                    # as the string itself
                    assert sc_item!=string           
                    # calculate edit distance using Damerau-
                    # Levenshtein distance
                    item_dist = dameraulevenshtein(sc_item, string)
                    
                    if item_dist<=max_edit_distance:
                        # should already be in dictionary if in
                        # suggestion list
                        assert sc_item in dictionary  
                        # trim list to contain state space
                        if (dictionary[q_item][1]>0): 
                            suggest_dict[sc_item] = \
                                (dictionary[sc_item][1], item_dist)
        
        # now generate deletes (e.g. a substring of string or of a
        # delete) from the queue item as additional items to check
        # -- add to end of queue
        assert len(string)>=len(q_item)
        if (len(string)-len(q_item))<max_edit_distance \
            and len(q_item)>1:
            for c in range(len(q_item)): # character index        
                word_minus_c = q_item[:c] + q_item[c+1:]
                if word_minus_c not in q_dictionary:
                    queue.append(word_minus_c)
                    # arbitrary value to identify we checked this
                    q_dictionary[word_minus_c] = None

    # return list of suggestions: (correction, edit distance)
    
    # only include words that have appeared a minimum number of times
    # note: make sure that we do not lose the original word
    as_list = [i for i in suggest_dict.items() 
               if (i[1][0]>min_count or i[0]==string)]
    
    # only include the most likely suggestions (based on frequency
    # and edit distance from original word)
    trunc_as_list = sorted(as_list, 
            key = lambda (term, (freq, dist)): (dist, -freq))[:max_sug]
    
    if len(trunc_as_list)==0:
        # to ensure Viterbi can keep running
        # -- use the word itself if no corrections are found
        return [(string, 0)]
        
    else:
        # drop the word frequency - not needed beyond this point
        return [(i[0], i[1][1]) for i in trunc_as_list]

    '''
    Output format:
    get_suggestions('file', dictionary)
    [('file', 0), ('five', 1), ('fire', 1), ('fine', 1), ('will', 2),
    ('time', 2), ('face', 2), ('like', 2), ('life', 2), ('while', 2)]
    '''
    
def get_emission_prob(edit_dist, poisson_lambda=0.01):
    '''
    The emission probability, i.e. P(observed word|intended word)
    is approximated by a Poisson(k, l) distribution, where 
    k=edit distance between the observed word and the intended
    word and l=0.01.
    
    Both the overall approach and the parameter of l=0.01 are based on
    the 2015 lecture notes from AM207 Stochastic Optimization.
    Various parameters for lambda between 0 and 1 were tested, which
    confirmed that 0.01 yields the most accurate word suggestions.
    '''
    
    return math.log(poisson.pmf(edit_dist, poisson_lambda))

######################
#
# Multiple helper functions are used to avoid KeyErrors when
# attempting to access values that are not present in dictionaries,
# in which case the previously specified default value is returned.
#
######################

def get_start_prob(word, start_prob, default_start_prob):
    '''
    P(word being at the beginning of a sentence)
    '''
    try:
        return start_prob[word]
    except KeyError:
        return default_start_prob
    
def get_transition_prob(cur_word, prev_word, 
                        transition_prob, default_transition_prob):
    '''
    P(word|previous word)
    '''
    try:
        return transition_prob[prev_word][cur_word]
    except KeyError:
        return default_transition_prob

def get_path_prob(prev_word, prev_path_prob):
    '''
    P(previous path)
    '''
    try:
        return prev_path_prob[prev_word]
    except KeyError:
        return math.log(math.exp(min(prev_path_prob.values()))/2.)  
    
def viterbi(words, dictionary, start_prob, default_start_prob, 
            transition_prob, default_transition_prob, max_edit_distance):
    '''
    Determine the most likely (intended) sequence, based on the
    observed sequence. Full details in preamble above.
    '''

    V = [{}]
    path = {}
    path_context = []
    
    # character level correction - used to determine state space
    corrections = get_suggestions(words[0], dictionary, max_edit_distance)
        
    # Initialize base cases (first word in the sentence)
    for sug_word in corrections:
        
        # compute the value for all possible starting states
        V[0][sug_word[0]] = math.exp(
            get_start_prob(sug_word[0], start_prob, 
                           default_start_prob)
            + get_emission_prob(sug_word[1]))
        
        # remember all the different paths (only one word so far)
        path[sug_word[0]] = [sug_word[0]]
 
    # normalize for numerical stability
    path_temp_sum = sum(V[0].values())
    V[0].update({k: math.log(v/path_temp_sum) 
                 for k, v in V[0].items()})
    
    # keep track of previous state space
    prev_corrections = [i[0] for i in corrections]
    
    # return if the sentence only has one word
    if len(words) == 1:
        path_context = [max(V[0], key=lambda i: V[0][i])]
        return path_context

    # run Viterbi for all subsequent words in the sentence
    for t in range(1, len(words)):

        V.append({})
        new_path = {}
        
        # character level correction
        corrections = get_suggestions(words[t], dictionary, max_edit_distance)
 
        for sug_word in corrections:
        
            sug_word_emission_prob = get_emission_prob(sug_word[1])
            
            # compute the probabilities associated with all previous
            # states (paths), only keep the maximum
            (prob, word) = max(
                (get_path_prob(prev_word, V[t-1]) 
                + get_transition_prob(sug_word[0], prev_word, 
                    transition_prob, default_transition_prob)
                + sug_word_emission_prob, prev_word) 
                               for prev_word in prev_corrections)

            # save the maximum probability for each state
            V[t][sug_word[0]] = math.exp(prob)
            
            # store the full path that results in this probability
            new_path[sug_word[0]] = path[word] + [sug_word[0]]
        
        # normalize for numerical stability
        path_temp_sum = sum(V[t].values())
        V[t].update({k: math.log(v/path_temp_sum) 
                     for k, v in V[t].items()})
        
        # keep track of previous state space
        prev_corrections = [i[0] for i in corrections]
 
        # don't need to remember the old paths
        path = new_path
     
    # after all iterations are completed, look up the word with the
    # highest probability
    (prob, word) = max((V[t][sug_word[0]], sug_word[0]) 
                       for sug_word in corrections)

    # look up the full path associated with this word
    path_context = path[word]

    return path_context

def get_count_mismatches(sentences):
    '''
    Helper function: compares the original sentence with the sentence
    that has been suggested by the Viterbi algorithm, and calculates
    the number of words that do not match.
    '''
    orig_sentence, sug_sentence = sentences
    count_mismatches = len([(orig_sentence[i], sug_sentence[i]) 
            for i in range(len(orig_sentence))
            if orig_sentence[i]!=sug_sentence[i]])
    return count_mismatches, orig_sentence, sug_sentence

def correct_document_context_parallel_naive(fname, dictionary,
                             start_prob, default_start_prob,
                             transition_prob, default_transition_prob,
                             max_edit_distance=3, num_partitions=6,
                             display_results=False):
    
    '''
    Load a text file and spell-check each sentence using the
    dictionary and probability tables that were created in the
    pre-processing stage.

    Suggested corrections are either printed to the screen or
    saved in a log file, depending on the settings.
    '''

    # note: various partitioning strategies were attempted for this
    # function, but they failed to yield significant improvements in
    # performance at any file size.

    ############
    #
    # load file & initial processing
    #
    ############
    
    # http://stackoverflow.com/questions/22520932/python-remove-all-non-alphabet-chars-from-string
    regex = re.compile('[^a-z ]')

    # broadcast Python dictionaries to workers (from pre-processing)
    bc_dictionary = sc.broadcast(dictionary)
    bc_start_prob = sc.broadcast(start_prob)
    bc_transition_prob = sc.broadcast(transition_prob)
    
    # load file contents and convert into one long sequence of words
    # RDD format: 'line 1', 'line 2', 'line 3', ...
    make_all_lower = sc.textFile(fname) \
            .map(lambda line: line.lower()) \
            .filter(lambda x: x!='')
        
    # split into individual sentences and remove other punctuation
    # RDD format: [words of sentence 1], [words of sentence 2], ...
    # cache because this RDD is used in multiple operations 
    split_sentence = make_all_lower.flatMap(lambda 
        line: line.replace('?','.').replace('!','.').split('.')) \
             .map(lambda sentence: regex.sub(' ', sentence)) \
             .map(lambda sentence: sentence.split()) \
             .filter(lambda x: x!=[]).cache()
    
    # use accumulator to count the number of words checked
    accum_total_words = sc.accumulator(0)
    split_sentence.flatMap(lambda x: x) \
            .foreach(lambda x: accum_total_words.add(1))
    
    # assign a unique id to each sentence
    # RDD format: (0, [words of sentence1]), (1, [words of sentence2]), ...
    # cache here after completing transformations - results in 
    # improvements in runtime that scale with file size
    # partition as sentence id will remain the key going forward
    sentence_id = split_sentence.zipWithIndex().map(
        lambda (k, v): (v, k)).partitionBy(num_partitions).cache()

    ############
    #
    # spell-checking
    #
    ############

    # use map operation to apply Viterbi algorithm to each sentence
    # RDD format: (0, [original sentence1], [corrected sentence1]),
    #             (1, [original sentence2], [corrected sentence2]), ...
    sentence_correction = sentence_id.mapValues(lambda v: (v, 
                viterbi(v, bc_dictionary.value, bc_start_prob.value, 
                        default_start_prob, bc_transition_prob.value, 
                        default_transition_prob, max_edit_distance)))
    ############
    #
    # output results
    #
    ############
    
    # count the number of corrections per sentence and drop any
    # sentences without suggested corrections
    # RDD format: 
    # (0, (corrections, [original sentence1], [corrected sentence1])),
    # (1, (corrections, [original sentence2], [corrected sentence2])), ...
    sentence_errors = sentence_correction.mapValues(lambda v: 
        (get_count_mismatches(v))). \
            filter(lambda (k, v): v[0]>0)
    
    # collect all sentences with identified errors (as list)
    sentence_errors_list = sentence_errors.collect()
    
    # count the number of potentially misspelled words
    num_errors = sum([s[1][0] for s in sentence_errors_list])
    
    # print suggested corrections
    if display_results:
        for sentence in sentence_errors_list:
            print 'Sentence %i: %s --> %s' % (sentence[0],
                ' '.join(sentence[1][1]), ' '.join(sentence[1][2]))
            print '-----'
    
    # output suggested corrections to file
    else:
        f = open('spell-log.txt', 'w')
        for sentence in sentence_errors_list:
            f.write('Sentence %i: %s --> %s\n' % (sentence[0], 
                ' '.join(sentence[1][1]), ' '.join(sentence[1][2])))
        f.close()
    
    print '-----'
    print 'Total words checked: %i' % accum_total_words.value
    print 'Total potential errors found: %i' % num_errors

def main(argv):
    '''
    Parse command line parameters (if any).

    Command line parameters are expected to take the form:
    -d : dictionary file
    -c : spell-checking file

    Default values are applied where files are not provided.
    https://docs.python.org/2/library/getopt.html
    '''

    # default values - use if not overridden
    dictionary_file = 'testdata/big.txt'
    check_file = 'testdata/yelp100reviews.txt'

    # read in command line parameters (if any)
    try:
        opts, args = getopt.getopt(argv,'d:c:',['dfile=','cfile='])
    except getopt.GetoptError:
        print 'contextSerial.py -d <dfile> -c <cfile>'
        print 'Default values will be applied.'

    # parse command line parameters    
    for opt, arg in opts:
        if opt in ('-d', '--dictionary'):
            dictionary_file = arg
        elif opt in ('-c', '--cfile'):
            check_file = arg

    # return command line parameters (or default values if not provided)
    return dictionary_file, check_file

if __name__ == '__main__':

    ############
    #
    # get input files and check that they are valid
    #
    ############

    # dictionary_file = used for pre-processing steps
    # check_file = text to be spell-checked
    dictionary_file, check_file = main(sys.argv[1:])

    dict_valid = os.path.isfile(dictionary_file)
    check_valid = os.path.isfile(check_file)

    if not dict_valid and not check_valid:
        print 'Invalid dictionary and spellchecking files. Could not run.'
        sys.exit()
    elif not dict_valid:
        print 'Invalid dictionary file. Could not run.'
        sys.exit()
    elif not check_valid:
        print 'Invalid spellchecking file. Could not run.'
        sys.exit()

    ############
    #
    # pre-processing
    #
    ############

    print 'Pre-processing with %s...' % dictionary_file

    start_time = time.time()

    dictionary, start_prob, default_start_prob, transition_prob, default_transition_prob = \
	    parallel_create_dictionary(dictionary_file)

    run_time = time.time() - start_time

    print '-----'
    print '%.2f seconds to run' % run_time
    print '-----'

    ############
    #
    # spell-checking
    #
    ############

    print 'Spell-checking %s...' % check_file

    start_time = time.time()

    correct_document_context_parallel_naive(check_file, dictionary,
                            start_prob, default_start_prob, 
                            transition_prob, default_transition_prob)

    run_time = time.time() - start_time

    print '-----'
    print '%.2f seconds to run' % run_time
    print '-----'

