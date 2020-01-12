"""
Author: Jaskaran Singh 
Start Date: Jan 11 2020
End Date: Jan 12 2020
Description: Bot going to Twitter to
References: https://gist.github.com/vickyqian/f70e9ab3910c7c290d9d715491cde44c
References: https://towardsdatascience.com/twitter-sentiment-analysis-classification-using-nltk-python-fa912578614c
DATASET: https://github.com/jbarnesspain/sota_sentiment/tree/master/datasets
https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
Latest Reference: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
"""
"________Importing Libraries___________"
import nltk
import tweepy
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import csv
import firebase_admin
nltk.download('stopwords')
stop_words = stopwords.words('english')
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
stem = PorterStemmer()
import pandas as pd
nltk.download('punkt')
import re, string
stop_words = stopwords.words('english')
import random
from firebase_admin import credentials, firestore
from nltk import classify
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import twitter_samples
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
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

train_tweets = pd.read_csv('train_tweets.csv')
test_tweets = pd.read_csv('test_tweets.csv')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = "Jatt is Jatt fuck you RBC no matter what"
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list)
freq_dist_pos = FreqDist(all_pos_words)

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

positive_dataset = [(tweet_dict, "P")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "N")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)
random.shuffle(dataset)
train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)

doc_ref = db.collection(u"application").document(u'zK4OS4dCy8XXvFDP2jAV')
hashTag = doc_ref.get("hashtagName")
print(u'hashtagName:{}'.format(hashTag.to_dict()))

print("Accuracy is:", classify.accuracy(classifier, test_data))
from random import randrange



while True:
    doc_ref = db.collection(u"application").document(u'zK4OS4dCy8XXvFDP2jAV')
    hashTag = doc_ref.get("hashtagName")
    print(u'hashtagName:{}'.format(hashTag.to_dict()))
    for tweet in tweepy.Cursor(api.search, q=str(hashTag), count=1, lang="en", since="2020-01-10").items():
        custom_tweet = tweet.text
        custom_tokens = remove_noise(word_tokenize(custom_tweet))
        print(tweet.text)
        print(classifier.classify(dict([token, True] for token in custom_tokens)))
        if classifier.classify(dict([token, True] for token in custom_tokens)) == "P":
            doc_ref = db.collection(u"twitter").add({
                u'Date': tweet.created_at,
                u'phrase': tweet.text,
                u'positive': True,
                u'technical': False
            })
        elif classifier.classify(dict([token, True] for token in custom_tokens)) == "N":
            doc_ref = db.collection(u"twitter").add({
                u'Date': tweet.created_at,
                u'phrase': tweet.text,
                u'positive': False,
                u'technical': False
            })

        tweetWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])

"""
def text_processing(tweet):
    # Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)

    new_tweet = form_sentence(tweet)

    # Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess

    no_punc_tweet = no_user_alpha(new_tweet)

    # Normalizing the words in tweets
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word, 'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet

    return normalization(no_punc_tweet)


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


for tweet in tweepy.Cursor(api.search,q="#RBC",count=1, lang="en", since="2020-01-10").items():
    #print(tweet.text)
    #print(text_processing(tweet.text))

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    tweetWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])


"""

