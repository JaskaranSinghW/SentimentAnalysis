"""
Author: Jaskaran Singh 
Start Date: Jan 11 2020
End Date: Jan 12 2020
Description: Bot going to Twitter to
References: https://gist.github.com/vickyqian/f70e9ab3910c7c290d9d715491cde44c
"""
"________Importing Libraries___________"
import nltk
"nltk.download()"
from nltk.corpus import stopwords
import tweepy
import csv
import firebase_admin
from firebase_admin import credentials, firestore
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()
import pandas as pd
import sentimentmod as s

"_________END____________"


"________API CREDENTIALS___________"
"Twitter API User credentials"
consumer_key = "HBb2EdO1Rn8zSiGHS1j49rNvd"
consumer_secret = "MkhGPXcP6sxLgM5iXqA4jheYKeg5RwYXqWTPw1ncOybzylGroB"
access_token = "973025558006185984-RKIUNbjzWODFrbuj5VaMWG7u5cbMf9G"
access_token_secret = "xwpSNs67Y945OgpasZsSDrU1p9KBCYn7e8Vo76Q1RWLi2"

"FireStore User Credentials"
cred = credentials.Certificate("rbcanalytica-firebase-adminsdk-mjex8-bef3e1c772.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

"_________END____________"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
# Open/Create a file to append data
tweetFile = open('tweet.csv', 'a')
tweetWriter = csv.writer(tweetFile)
for tweet in tweepy.Cursor(api.search,q="#RBC",count=10, lang="en", since="2020-01-10").items():
    print (tweet.created_at, tweet.text)
    "Creating Tokens from Entered Text"
    tokens = nltk.word_tokenize(tweet.text)
    "Tagging text (Nown,Pronon,verb..etc)"
    tagged = nltk.pos_tag(tokens)
    tagged[0:6]
    "Removing StopWords from input"
    stop_words = set(stopwords.words("english"))
    filteredWords = []
    for w in tokens:
        if w not in stop_words:
            filteredWords.append(w)

    stemmed_words = []
    ps = PorterStemmer()
    for w in filteredWords:
        stemmed_words.append(ps.stem(w))
    sentiment_value, confidence = s.sentiment(stemmed_words)
    print(tweet, sentiment_value, confidence)

    if confidence * 100 >= 80:
        output = open("twitter-out.txt", "a")
        output.write(sentiment_value)
        output.write('\n')
        output.close()

    tweetWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

#doc_ref = db.collection(u"twitter").document(u"#1")
#doc_ref.set({
 #   u'phrase': u'RBC application is working GREAT',
  #  u'positive': True,
   # u'technical': True
#})
