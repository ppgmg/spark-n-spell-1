# spark-n-spell README

Check out our website at http://www.spark-n-spell.com

## EXECUTABLES

**Note:** All scripts were developed and tested using Python 2.7 and spark-1.5.0 and may not run as expected on other configurations.

### Word-Level Correction

(a) Run our Python port of SymSpell to correct *individual* words.

- download *symspell_python.py* to your local directory
- download the dictionary file *big.txt* to the same directory, from this github repository or one of the following additional sources: 
  - https://github.com/dominedo/spark-n-spell/tree/master/testdata
  - http://norvig.com/ngrams/ 
  - s3n://spark-n-spell/big.txt 
  - (or use your own dictionary file renamed as *big.txt*)
- at the prompt, run:  `python symspell_python.py`
- type in the word to be corrected at the interactive prompt

(b) Run our Spark program to correct *individual* words.

- download *word_correct_spark.py* to your local directory (you should have Spark 1.5.0 installed, and you must be able to call spark-submit from that directory)
- if not already done, download the dictionary file *big.txt* from one of sources listed above
- at the prompt, run:  `spark-submit word_correct_spark.py -w "<word to correct>"` 
  - e.g. `spark-submit word_correct_spark.py -w "cvhicken"`

(c) Run our word-level Spark document checker.

Note this will be fairly slow as the current version internally generates all possible suggestions for each word in the test document. For a faster document checker, please run one of the context-level spellcheckers below.

- download *word_level_doc_correct.py* to your local directory (you should have Spark 1.5.0 installed, and you must be able to call spark-submit from that directory)
- if not already done, download the dictionary file *big.txt* from one of sources listed above
- download a document file to be tested into the working directory (some example test files of varying sizes can be found at https://github.com/dominedo/spark-n-spell/tree/master/testdata)
  - you will need to specify the name of this file when executing the python script as noted below using -c (otherwise, by default the script will look for a test document named *test.txt* in the working directory)
- at the prompt, run:  `spark-submit word_level_doc_correct.py -c "<.txt file to check>"`
  - e.g. `spark-submit word_level_doc_correct.py -c "test.txt"`
  - optionally, you may add a *-d file.txt* argument to specify a different dictionary file
  - corrections are logged to *log.txt* in the local directory

### Context-Level Correction (Viterbi algorithm)

(a) Run our Python implementation of context-level document checking.

- download *contextSerial.py* to your local directory
- if not already done, download the dictionary file *big.txt* from one of sources listed above
- download one of our sample test files from the *testdata* sub-folder, or prepare a .txt file of your own for checking
- at the prompt, run:  `python contextSerial.py`
- use the custom parameter -d to override the default dictionary (*big.txt*) and/or the custom parameter -c to override the default document for checking (*yelp100reviews.txt*)
    - e.g.  `python contextSerial.py -d 'mycustomdictionary.txt' -c 'mycustomdocument.txt'`

(b) Run our naive SPARK implementation of context-level checking.

- download *contextSPARKnaive.py* to your local directory
- if not already done, download the dictionary file *big.txt* from one of sources listed above
- download one of our sample test files from the *testdata* sub-folder, or prepare a .txt file of your own for checking
- at the prompt, run:  `spark-submit contextSPARKnaive.py`
- use the custom parameter -d to override the default dictionary (*big.txt*) and/or the custom parameter -c to override the default document for checking (*yelp100reviews.txt*)
    - e.g.  `spark-submit contextSPARKnaive.py -d 'mycustomdictionary.txt' -c 'mycustomdocument.txt'`

(c) Run our full SPARK implementation of context-level checking.

- download *contextSPARKfull.py* to your local directory
- if not already done, download the dictionary file *big.txt* from one of sources listed above
- download one of our sample test files from the *testdata* sub-folder, or prepare a .txt file of your own for checking
- at the prompt, run:  `spark-submit contextSPARKfull.py`
- use the custom parameter -d to override the default dictionary (*big.txt*) and/or the custom parameter -c to override the default document for checking (*yelp100reviews.txt*)
    - e.g.  `spark-submit contextSPARKfull.py -d 'mycustomdictionary.txt' -c 'mycustomdocument.txt'`

## DOCUMENTATION

Consult our IPYTHON NOTEBOOKS for documentation on our coding and testing process.

- For word-level correction:  *word_level_documentation.ipynb*
  
- For context-level correction:  *context_level_documentation.ipynb* 
  
  In order to view all related content and run the code, both files require the *img* , *sample* , and *testdata* sub-directories.

### OTHER DOCUMENTS IN THIS REPOSITORY

This repository also includes the following, for reference (see iPython Notebooks for details):

- *other_versions* folder:
  
  - *serial_listsugg.py* : word-level, word checker, serial, no early termination
  - *serial_single.py* : word-level, word checker, serial, same as *symspell_python.py*
  - *serial_document.py* : word-level, document checker, serial
  - *spark_1.py* : word-level, word checker, slow SPARK version
  - *spark_2.py* : word-level, word checker, faster SPARK version
  - *spark_3.py* : word-level, word checker, also fast SPARK version, same as *word_correct_spark.py*
  - *spark_4.py* : word-level, document checker, SPARK, same as *word_level_doc_correct.py*
  - *contextSPARKapproximate.py* : context-level, document-checker (warning: memory requirements grow exponentially with the size of the problem; only run on very small files e.g. *test.txt*)
  
- *testdata* folder: (all files also available at s3n://spark-n-spell/)
  
  - *big.txt* (6.5MB): used to create the dictionary and probability tables, where appropriate (source: http://norvig.com/ngrams/)
  - *test.txt* (106 bytes): variations of "this is a test"; used for early development and testing
  - *yelp1review.txt* (1KB): 1 Yelp restaurant review (183 words)
  - *yelp10reviews.txt* (8KB): 10 Yelp restaurant reviews (1,467 words)
  - *yelp100reviews.txt* (65KB): 100 Yelp restaurant reviews (12,029 words)
  - *yelp250reviews.txt* (173KB): 250 Yelp restaurant reviews (32,408 words)
  - *yelp500reviews.txt* (354KB): 500 Yelp restaurant reviews (66,602 words)
  - *yelp1000reviews.txt* (702KB): 1,000 Yelp restaurant reviews (131,340 words)