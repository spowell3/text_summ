

# Purpose of this script:
# To provide intro into Python-based text analytics methods

# Step one Learn and setup pip

import os #used for filepaths and such
import time # used just to calc run time on diff functions
import pandas as pd
import pickle
import json # this package helps make json prettier and easier to handle than a dictionary
from gensim.summarization import summarize # gensim

start_time = time.time()
check_time = start_time

def time_check(start_t, check_t):
    print("--- %s seconds since START---" % round(time.time() - start_t,3))
    print("--- %s seconds since CHECKED---" % round(time.time() - check_t,3))
    return check_t

# Dancing between directories
#start_path = os.getcwd()
#path = "C:/Users/Documents/wherever you stashed a file of interest"
#os.chdir(path)
#os.chdir(start_path)

# Importing smaller, more accessible df (from pickled file)

with open('json_tiny.json') as jfile:
    json_tiny = json.load(jfile)

print(json.dumps(json_tiny, sort_keys=True, indent=4, separators=(',', ': '))) # Harder to write, but easier to read.

df_tiny = pd.read_pickle('df_tiny')

print(df_tiny.loc[1099, 'summary'])

print(df_tiny.loc[1099,'coverage'])

len(df_tiny.loc[1244]['text'])/len(df_tiny.loc[1244]['summary']) # These numbers are similar...

print(df_tiny.loc[1265,'compression'])  # ... but not exactly equal... Interesting.

print(df_tiny.loc[1044, "density_bin"]) # TODO Research what defines 'mixed'




##############################################################################
# region TOPIC MODELLING
# TODO Explore gensim and alternate packages for techniques to try
# endregion


##############################################################################
# region TEXT SUMMARIZATION METHODS

# a function to print different summaries for comparison
def summ_compare(row):
    # print("DENSITY: ",row['density']," DATE: ", row['date'])
    if row['summary'] == None:
        summary = ""
    else:
        summary = row['summary']
    print("SOURCE SUMMARY: " + summary)
    print()
    print("TextRank SUMMARY: " + row['TextRank_summary'])
    print("\n")


# a function to catch any errors thrown by the summarize function
# because sometimes it doesn't like the short # of sentences in 'text'
def summ_catch(text):
    try:
        # Text Rank Algorithm... https://github.com/kedz/sumpy also has textrank
        TextRank = summarize(text, word_count=40)
        return (TextRank)
    except ValueError:
        print('Oops')
        return ('NaN')

#applying the summarization function to our tiny database

df_tiny['TextRank_summary']=df_tiny['text'].apply(summ_catch)
df_tiny.loc[:,['summary','TextRank_summary']] # hard to read
df_tiny.apply(summ_compare, axis=1) # less hard to read

# Outputting file as pickled df and json
df_tiny.to_pickle('df_tiny') # saving for easy access by all without thin library
with open('json_tiny.json', 'w') as outfile:
    json.dump(json_tiny, outfile)

# Algorithms ordered here by performance on Summari.es
# TODO Algorithm #1: Lede-3 Baseline (Grusky et al., 2018) Extractive
#   https://github.com/kedz/sumpy has Lede-3,
#       last commit was 3 yrs ago
#       doesn't seem considerate of computation time
# TODO Algorithm #2: Pointer-Generator  (See et al., 2017) Mixed
#   https://github.com/abisee/pointer-generator
#
# TODO Algorithm #4: Seq2Seq + Attention (Rush et al., 2015) Abstractive (notably low performance)
#
# TODO Understand the capabilities of following:
#   https://www.quora.com/When-is-better-to-use-NLTK-vs-Sklearn-vs-Gensim
#   https://github.com/RaRe-Technologies/gensim
#       very cognisant of computation time
#       also heavy topic modelling capabilities
#   https://github.com/dipanjanS/text-analytics-with-python/tree/master/Old_Edition_v1/notebooks/Ch05_Text_Summarization
#       Dependent on gensim
#       Python 2
#
# TODO Identify additional/better algorithms
# endregion


##############################################################################
#region SCORE SUMMARIES

# TODO Score summaries against provided summaries
# ROUGE
#   https://github.com/kavgan/ROUGE-2.0
#   https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5
#

#endregion

check_time = time_check(start_time, check_time)