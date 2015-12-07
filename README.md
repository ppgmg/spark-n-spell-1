# spark-n-spell README

## EXECUTABLES

### Word-Level Correction

(a) Run our Python port of SymSpell to correct *individual* words.

- download *symspell_python.py* to your local directory
- download the dictionary file *big.txt* to the same directory, from one of the following sources: 
    - https://github.com/dominedo/spark-n-spell/tree/master/testdata
    - http://norvig.com/ngrams/ 
    - s3n://spark-n-spell/big.txt 
    - (or use your own dictionary file renamed as big.txt)
- at the prompt, run:  python symspell_python.py
- type in the word to be corrected at the interactive prompt

(b) Run our word-level Spark document checker.

Note this will be fairly slow as the current version internally generates all possible suggestions for each word in the test document. For a faster document checker, please run one of the context-level spellcheckers below.

- download *word_level_doc_correct.py* to your local directory (you should have Spark 1.5.0 installed, and you must be able to call spark-submit from that directory)
- if not already done, download the dictionary file *big.txt* from one of sources listed above
- download a document file to be tested (e.g. *test.txt*) into the working directory (some examples are in https://github.com/dominedo/spark-n-spell/tree/master/testdata)
    - you will need to specify the name of this file when executing the python script as noted below using -c (otherwise, by default the script will look for a test document named *test.txt* in the working directory)
- at the prompt, run:  spark-submit word_level_doc_correct.py -c "<.txt file to check>"
    - optionally, you may add a *-d file.txt* argument to specify a different dictionary file
    - corrections are logged to "log.txt" in the local directory

### Context-Level Correction

(a) Run ... some serial python version ...
(b) Run ... some spark version ...
(c) Run ... some other spark version ...

## DOCUMENTATION

Consult our IPYTHON NOTEBOOKS for documentation on our coding and testing process.

- For word-level correction:  word_level_documentation.ipynb  (requires *times1.png* and *times2.png* in same directory)
- For context-level correction: ... add ipython notebook ...

### OTHER DOCUMENTS IN THIS REPOSITORY

This repository also includes the following, for reference (see IPython Notebooks for details):

- *other_versions* folder:
    - serial_listsugg.py (word-level, word checker, serial, no early termination)
    - serial_single.py (word-level, word checker, serial, same as symspell_python.py)
    - serial_document.py (word-level, document checker, serial)
    - spark_1.py (word-level, word checker, slow SPARK version)
    - spark_2.py (word-level, word checker, faster SPARK version)
    - spark_3.py (word-level, word checker, also fast SPARK version)
    - spark_4.py (word-level, document checker, SPARK, same as word_level_doc_correct.py)
    - .... other versions of context-level ...

- *testdata* folder:
    - ... description of test files ...


