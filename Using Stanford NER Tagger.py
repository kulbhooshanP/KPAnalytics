# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 14:42:36 2016

@author: kulpatil
"""
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from __future__ import division
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim import corpora,models
from operator import itemgetter
import csv, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Phrases
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
import nltk
lemma = WordNetLemmatizer()

tokenizer = RegexpTokenizer(r'\w+')

st = StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner/stanford-ner-3.6.0.jar')

print st.tag('"Preference for #fair skin reinforced in #movies, #television programmes & especially #advertising" buff.ly/2fW5ivc #DarkIsBeautiful pic.twitter.com/oeoPVKitS0'.split()) 

st_POS = StanfordPOSTagger('stanford-postagger/models/english-bidirectional-distsim.tagger',
                       'stanford-postagger/stanford-postagger-3.6.0.jar') 
print st_POS.tag('What is the airspeed of an unladen swallow ?'.split()) 


data=["hello world, ha ha India . i work in Capgemini",
"Preference for #fair skin reinforced in #movies, #television programmes & especially #advertising"]

for row in data:
    tokens = tokenizer.tokenize(row)
    ner=st.tag(tokens)
    for token, tag in ner:
        if tag=='LOCATION':
            print token,tag
#st.tag_sents(tokenized_sents)
for row in data:
    tokens = tokenizer.tokenize(row)
    pos=st_POS.tag(tokens)
    for token, tag in pos:
        if ('NN' in tag) or ('VB' in tag ) :
            print token,tag
