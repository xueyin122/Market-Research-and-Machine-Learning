#!/usr/bin/env python
# coding: utf-8

# pip install tweepy
# !pip install pyLDAvis
# !pip install spacy-langdetect
# !pip install spacy
# !pip install spacy==3.4.4


# In[2]:


import tweepy
import pandas as pd 
import os
import itertools
import collections
import re
import nltk
#nltk.download('punkt')
from nltk import everygrams, word_tokenize
from nltk.corpus import stopwords

import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint

import pyLDAvis
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()
import pickle 

from spacy_langdetect import LanguageDetector
import spacy

from spacy.language import Language

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel


# In[3]:


# Variables that contains the credentials to access Twitter API
ACCESS_TOKEN = '904181285752639488-q1X0fsHyxQeLbkjoeZo5WoVtof9mB2j' 
ACCESS_SECRET = 'NN0dueO12GxwIXs4m9jV2GsZH2OVddOup5mt1lxWggrw3' 
CONSUMER_KEY = 'kMbYXac7rcgngptMelbCITu9m'
CONSUMER_SECRET = 'ECGStWdeeMRBaVVxZi1mYil31j4e9RoSkEiz7Uq5TRTMKA5eCT'

BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAHiXlAEAAAAASR22rJVlGfapOnQ6Nb0I%2BRaZGFo%3DDv0v57y7jQ95IgXohManxD7qlh8ObDOOkProdlhseMxATlNyl5'

# Setup access to API
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth)
    return api


# Create API object
api = connect_to_twitter_OAuth()


# In[4]:


# fuction to extract data from tweet object
def extract_tweet_attributes(tweet_object):
    # create empty list
    tweet_list =[]
    # loop through tweet objects
    for tweet in tweet_object:
        tweet_id = tweet.id # unique integer identifier for tweet
        text = tweet.text # utf-8 text of tweet
        favorite_count = tweet.favorite_count
        retweet_count = tweet.retweet_count
        created_at = tweet.created_at # utc time tweet created
        source = tweet.source # utility used to post tweet
        reply_to_status = tweet.in_reply_to_status_id # if reply int of orginal tweet id
        reply_to_user = tweet.in_reply_to_screen_name # if reply original tweetes screenname
        retweets = tweet.retweet_count # number of times this tweet retweeted
        favorites = tweet.favorite_count # number of time this tweet liked
        # append attributes to list
        tweet_list.append({'tweet_id':tweet_id, 
                          'text':text, 
                          'favorite_count':favorite_count,
                          'retweet_count':retweet_count,
                          'created_at':created_at, 
                          'source':source, 
                          'reply_to_status':reply_to_status, 
                          'reply_to_user':reply_to_user,
                          'retweets':retweets,
                          'favorites':favorites})
    # create dataframe   
    df = pd.DataFrame(tweet_list, columns=['tweet_id',
                                           'text',
                                           'favorite_count',
                                           'retweet_count',
                                           'created_at',
                                           'source',
                                           'reply_to_status',
                                           'reply_to_user',
                                           'retweets',
                                           'favorites'])
    return df


# In[6]:


client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Get tweets that contain the hashtag #petday
# lang:en is asking for the tweets to be in english
query = "activision blizzard lang:en"
tweets = tweepy.Paginator(client.search_recent_tweets, query=query,
                              tweet_fields=['context_annotations', 'created_at', 'text', 'id'], max_results=100).flatten(limit=1000)

tweet_list =[]

for tweet in tweets:
    tweet_id = tweet.id
    text = tweet.text
    tweet_list.append({'tweet_id':tweet_id, 'text':text})
    df = pd.DataFrame(tweet_list, columns=['tweet_id', 'text'])
df


# In[8]:


# Remove URLs, remove at mentions
def remove_url(txt):
    return " ".join(re.sub("https://t.co/\w+", "", txt).split())

def remove_atmention(txt):
    return " ".join(re.sub("@\w+[^\s]", "", txt).split())

all_tweets_no_urls = [remove_url(tweet) for tweet in df['text']]
all_tweets_no_atmention = [remove_atmention(tweet) for tweet in all_tweets_no_urls]


# In[11]:


words_in_tweet = [tweet.lower().split() for tweet in all_tweets_no_atmention]

all_words_no_urls = list(itertools.chain(*words_in_tweet))

# Create counter
counts_no_urls = collections.Counter(all_words_no_urls)

clean_tweets_no_urls = pd.DataFrame(counts_no_urls.most_common(30),
                             columns=['words', 'count'])


# In[13]:


#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))
stop_words = stopwords.words('english')
stop_words.extend(['activision', 'blizzard', 'activision/blizzard', 'blizzard,', 'rt', 'RT'])


# In[14]:


tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]
all_words_nsw = list(itertools.chain(*tweets_nsw))

counts_nsw = collections.Counter(all_words_nsw)

# Most common 15 words
counts_nsw.most_common(15)


# In[15]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]


# In[16]:


data_words = remove_stopwords(all_words_nsw)


# In[17]:


# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# In[28]:


# number of topics
num_topics = 3
# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[29]:


# Visualize the topics
pyLDAvis.enable_notebook()
LDAvis_data_filepath = 'lda.html'
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, 'lda.html')
LDAvis_prepared


# In[21]:


# Get coherence score (method = c_v)
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[22]:


# # Get coherence score (method = u_mass)
coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[23]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[25]:


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=2, limit=40, step=6)
# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

