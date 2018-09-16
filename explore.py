##############################################################################
#region DOWNLOAD GUIDANCE
# ^^ note the region tags and how they work with your IDE

#
# generally followed instructions from https://github.com/clic-lab/newsroom
#
# 0. Learn and setup pip
#   pip: it's super valuable. Similar to install.packages() function of R, just a bit less intuitive. Get on-board the pip-train
#      if you ever hit errors during a pip install, I try to address the error message, then re-run
#      don't start manually moving files
#
# 1. Install Microsoft Studio Build Tools
#     otherwise, during step 2, you will error specifying need for this software
#     I have no idea what it is, but I installed it from link below
#     https://visualstudio.microsoft.com/visual-cpp-build-tools/
#     Note: link provided by pip in terminal was broken
#     4.8 GB!!! took ~40 min
#
# 2. Download newsroom
#   terminal input: pip install -e git+git://github.com/clic-lab/newsroom.git#egg=newsroom
#   Will likely install a TON of packages
#   took 5-10 min
#
# 3. Download wget and get initial files
#   download wget zip file from https://eternallybored.org/misc/wget/
#   Extract zip
#   cmd prompt: use `cd` and `dir` to nav to extracted wget-1.19.4 folder
#   cmd prompt: `wget https://summari.es/files/thin.tar` Took 20 min
#   cmd prompt: `tar xvf thin.tar`
#   Now located in Program files\wget-1.19.4 folder
#
# 4. scraping process
#   moved thin and newsroom directories to own folder (titled "text_sum")
#   cmd prompt: `newsroom-scrape --thin thin/dev.jsonl.gz --archive dev.archive`
#   2 GB, ~10.5 hrs to download!!!
#
# 5. extraction process
#   cmd prompt: `newsroom-extract --archive dev.archive --dataset dev.data`
#   ~12 hrs to extract!!!
#   Should now be ready to start. Finally.
#
#
# Failed attempts: ***************
# X. Direct browser download of thin.tar from https://summari.es/download/ did not work
#   Large file size? I got repeated 'network errors'
# X. pip installed wget
#   package attempt to dl didn't work
#     import wget
#     thin_url = "https://summari.es/files/thin.tar"
#     thin_dest = "C:/Users/Steven/Documents/MSA/Practicum/text_sum/thin"
#     wget.download(thin_url, thin_dest) # Does not work
#
# X.http://noahcoad.com/post/614/using-wget-on-windows
#   But couldn't find bin file
#   Saved in C:\Users\Steven\Downloads\wget-1.19, now deleted
#
# X. downloaded wget from https://sourceforge.net/projects/gnuwin32/files/wget/1.9-1/
#   Old, disfunctional version?
#
# Possible problems moving forward: *****************
#   Location of newsroom download after pip install C:\Users\Steven\src
#   Location where package should be?
#       C:\Users\Steven\AppData\Local\Programs\Python\Python36-32\Lib\site-packages\newsroom.egg-link
#       ... or in local dir
#endregion

#pip install pandas if you don't have it
import os #used for filepaths and such
import time # used just to calc run time on diff functions
import pandas as pd
from newsroom import jsonl
import pickle


##############################################################################
#region   ACCESSING ARTICLES
#######################################

start_time = time.time()

start_path = os.getcwd()

path = "C:/Users/Steven/Documents/MSA/Practicum/text_sum/"

# Read entire file:

# must first set working directory to location of thin database
os.chdir("C:/Users/Steven/Documents/MSA/Practicum/thin")
with jsonl.open("dev.data", gzip = True) as train_file:
    train = train_file.read()
os.chdir(start_path)

# WORKING CODE, but commented out cause it'd easier to view as df below
# Read file entry by entry:
# with jsonl.open("dev.data", gzip = True) as train_file:
#     i=0
#     for entry in train_file:
#         print("SUMMARY: ", entry["summary"])
#         print("ENTRY:\n", entry["text"])
#         print('###############################')
#         i += 1
#         if i == 30:
#             print("--- %s seconds ---" % (time.time() - start_time))
#             break

# Notes:
#     30 entry took ~18 secs to print summary and text
#     All 108,730 entries took ~120 seconds
#     108,730 articles are stored in a list
#     Each article is stored as a JSON object (technically a dictionary in Python)
#     JSON is annoying, but common for APIs and web nonsense

# access just a values of a single article

print(train[10344]) # notice that there's a lot of info (url, date, summary, density?, compression, title)
print(train[10344]['summary'])

# Q: Where does summary column come from?
# A:
# "NEWSROOM’s summaries were written by authors
# and editors in the newsrooms of news, sports,
# entertainment, financial, and other publications.
# The summaries were published with articles as
# HTML metadata ...
# NEWSROOM summaries are written by humans, for common
# readers, and with the explicit purpose of summarization.
# As a result, NEWSROOM is a nearly two
# decade-long snapshot representing how singledocument
# summarization is used in practice across
# a variety of sources, writers, and topics."
#
# TL;DR THIS DATABASE IS AWESOME!!!!!!

train[8999]['coverage'] # TODO Find Coverage Formula in paper

len(train[10344]['text'])/len(train[10344]['summary']) # These numbers are similar,
train[10344]['compression']  # ... but not exactly equal. Interesting.

train[10344]["density_bin"] # TODO Research what defined defines 'mixed'

import json # this package helps make json prettier and easier to handle than a dictionary

print(json.dumps(train[10344], sort_keys=True, indent=4, separators=(',', ': '))) # Harder to write, but easier to read.

#endregion


##############################################################################
#region     CLEANING
#######################################

# Converting into dataframe b/c it's hard to not think in dataframes, I suppose
# probably bad practice for memory usage purposes
df = pd.DataFrame(train[0:len(train)], range(0,len(train)))

# TODO (v low priority) Fix dates to be recognized as date type
#  df['date'] = df['date'].astype('datetime') #will need a better attempt

# TODO serialize text with pickle package into binary?

#endregion


##############################################################################
#region COMPARE ARTICLES
#######################################
# TODO Compare descriptive statistics of summari.es articles to large text set
# Sources
# Length
# Entities?
#endregion


##############################################################################
#region    APPLYING SUMMARIZATION   ###
#######################################

# again, pip install if you don't have it
from gensim.summarization import summarize

# a function to print different summaries for comparison
def summ_compare(row):
    #print("DENSITY: ",row['density']," DATE: ", row['date'])
    if row['summary'] == None:
        summary= ""
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
        return(TextRank)
    except ValueError:
        print('Oops')
        return('NaN')

# get memory error when trying to apply this to the whole df. Ugh.
# df['TextRank_summary']=df['text'].apply(summ_catch)

# slimming df down to a few rows to apply the TextRank function
df_tiny = df.loc[1000:1100,('summary','text')].copy() # copy avoids the case where changing df_tiny also changes df
df_tiny['TextRank_summary']=df_tiny['text'].apply(summ_catch)
df_tiny.loc[:,['summary','TextRank_summary']] # hard to read
df_tiny.apply(summ_compare, axis=1) # less hard to read


df_tiny.to_pickle('df_tiny') # saving for easy access by all without thin library
df_tiny =pd.read_pickle('df_tiny') # example of reading back in

# df['gensin_summary']=df.apply(lambda x: summ_catch(df['text']), axis=1)

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

#endregion


##############################################################################
#region  SCORE SUMMARIES
#######################################
# TODO Score summaries against provided summaries
# ROUGE
#   https://github.com/kavgan/ROUGE-2.0
#   https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5
#

# TODO Lingering Questions ?Q?
# Handling fake news. Outside of scope.
#endregion

print("--- %s seconds ---" % (time.time() - start_time))