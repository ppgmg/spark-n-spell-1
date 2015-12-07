# contextSerial.py - Serial implementation

import re
import math
from scipy.stats import poisson
import time
import sys, getopt
import os

######################
#
# Submission by Gioia Dominedo (Harvard ID: 40966234) for
# CS 205 - Computing Foundations for Computational Science
# 
# This is part of a joint project with Kendrick Lo that includes a
# separate component for word-level checking. This script includes 
# serial Python code for context-level spell-checking adapted
# from third party algorithms (Symspell and Viterbi algorithms). 
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
# More detail on the specific implementation is included below.
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

def create_dictionary_entry(w, dictionary, max_edit_distance):
    '''
    Add a word and its derived deletions to the dictionary.
    Dictionary entries are of the form:
    ([list of suggested corrections], frequency of word in corpus)
    '''

    new_real_word_added = False
    
    # check if word is already in dictionary
    if w in dictionary:
        # increment count of word in corpus
        dictionary[w] = (dictionary[w][0], dictionary[w][1] + 1)
    else:
        # create new entry in dictionary
        dictionary[w] = ([], 1)  
        
    if dictionary[w][1]==1:
        
        # first appearance of a word in the corpus
        # note: word may already be in dictionary as a derived word
        # (e.g. by deleting character from a real word) but the
        # word counter frequency is not incremented in those cases
        
        new_real_word_added = True
        deletes = get_deletes_list(w, max_edit_distance)
        
        for item in deletes:
            if item in dictionary:
                # add (correct) word to delete's suggested correction
                # list if not already there
                if item not in dictionary[item][0]:
                    dictionary[item][0].append(w)
            else:
                # note: frequency of word in corpus is not incremented
                dictionary[item] = ([w], 0)  
        
    return new_real_word_added

def pre_processing(fname, max_edit_distance=3):
    '''
    Load a text file and use it to create a dictionary and
    to calculate start probabilities and transition probabilities. 
    '''

    dictionary = dict()
    start_prob = dict()
    transition_prob = dict()
    word_count = 0
    transitions = 0
    
    with open(fname) as file:    
        
        for line in file:
            
            # process each sentence separately
            for sentence in line.replace('?','.').replace('!','.').split('.'):
                
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', sentence.lower())      
                
                for w, word in enumerate(words):
                    
                    # create/update dictionary entry
                    if create_dictionary_entry(
                        word, dictionary, max_edit_distance):
                            word_count += 1
                        
                    # update probabilities for Hidden Markov Model
                    if w == 0:

                        # probability of a word being at the
                        # beginning of a sentence
                        if word in start_prob:
                            start_prob[word] += 1
                        else:
                            start_prob[word] = 1
                    else:
                        
                        # probability of transitionining from one
                        # word to another
                        # dictionary format:
                        # {previous word: {word1 : P(word1|prevous
                        # word), word2 : P(word2|prevous word)}}
                        
                        # check whether prior word is present
                        # - create if not
                        if words[w - 1] not in transition_prob:
                            transition_prob[words[w - 1]] = dict()
                            
                        # check whether current word is present
                        # - create if not
                        if word not in transition_prob[words[w - 1]]:
                            transition_prob[words[w - 1]][word] = 0
                            
                        # update value
                        transition_prob[words[w - 1]][word] += 1
                        transitions += 1
                    
    # convert counts to log-probabilities, to avoid underflow in
    # later calculations (note: natural logarithm, not base-10)

    # also calculate (smalle) default probabilities for words that 
    # have not already been seen
    
    # probability of a word being at the beginning of a sentence
    total_start_words = float(sum(start_prob.values()))
    default_start_prob = math.log(1/total_start_words)
    start_prob.update( 
        {k: math.log(v/total_start_words)
         for k, v in start_prob.items()})
    
    # probability of transitioning from one word to another
    default_transition_prob = math.log(1./transitions)
    transition_prob.update(
        {k: {k1: math.log(float(v1)/sum(v.values()))
             for k1, v1 in v.items()} 
         for k, v in transition_prob.items()})

    # output summary statistics
    print 'Total unique words in corpus: %i' % word_count
    print 'Total items in dictionary: %i' \
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

def correct_document_context(fname, dictionary, 
                             start_prob, default_start_prob,
                             transition_prob, default_transition_prob,
                             max_edit_distance=3, display_results=False):
    
    '''
    Load a text file and spell-check each sentence using the
    dictionary and probability tables that were created in the
    pre-processing stage.

    Suggested corrections are either printed to the screen or
    saved in a log file, depending on the settings.
    '''

    doc_word_count = 0
    corrected_word_count = 0
    sentence_errors_list = []
    total_sentences = 0
    
    with open(fname) as file:
        
        for i, line in enumerate(file):
            
            for sentence in line.replace('?','.').replace('!','.').split('.'):
                
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', sentence.lower())  
                doc_word_count += len(words)
                
                if len(words) > 0:
                
                    # run Viterbi algorithm for each sentence and
                    # obtain most likely correction (may be the same
                    # as the original sentence)
                    suggestion = viterbi(words, dictionary,
                                start_prob, default_start_prob, 
                                transition_prob, default_transition_prob,
                                max_edit_distance)

                    # display sentences with suggested changes
                    if words != suggestion:
                        
                        # keep track of all potential errors
                        sentence_errors_list.append([total_sentences, 
                            (words, suggestion)])

                        # update count of corrected words
                        corrected_word_count += \
                        sum([words[j]!=suggestion[j] 
                             for j in range(len(words))])
                        
                    # used for display purposes
                    total_sentences += 1
  
    # print suggested corrections
    if display_results:
        for sentence in sentence_errors_list:
            print 'Sentence %i: %s --> %s' % (sentence[0],
                ' '.join(sentence[1][0]), ' '.join(sentence[1][1]))
            print '-----'
    
    # output suggested corrections to file
    else:
        f = open('spell-log.txt', 'w')
        for sentence in sentence_errors_list:
            f.write('Sentence %i: %s --> %s\n' % (sentence[0], 
                ' '.join(sentence[1][0]), ' '.join(sentence[1][1])))
        f.close()
            
    # display summary statistics
    print 'Total words checked: %i' % doc_word_count
    print 'Total potential errors found: %i' % corrected_word_count

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

    dictionary, start_prob, default_start_prob, \
    transition_prob, default_transition_prob \
    = pre_processing(dictionary_file)

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

    correct_document_context(check_file, dictionary,
                             start_prob, default_start_prob, 
                             transition_prob, default_transition_prob)

    run_time = time.time() - start_time

    print '-----'
    print '%.2f seconds to run' % run_time
    print '-----'


