"""
Author: Jaskaran Singh 
Start Date: DEC 17 2019 
End Date: Dec __ 2019
https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
Description: Take in Sentences from User Process it and output the sentiment

"""
import nltk
nltk.download('')
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

"Uset Input"
sentense = input("Enter Text to be Tested:")
print(sentense)
"Creating Tokens from Entered Text"
tokens = nltk.word_tokenize(sentense)
"print(tokens)"
"Tagging text (Nown,Pronon,verb..etc)"
tagged = nltk.pos_tag(tokens)
tagged[0:6]
"Creating Word Frequeency Graph"
frequencyDist = FreqDist(tagged)
frequencyDist.plot(30,cumulative=False)
"plt.show"
"Removing StopWords from imput"
stop_words=set(stopwords.words("english"))
filtereWords=[]
for w in tokens:
    if w not in stop_words:
        filtereWords.append(w)
print(filtereWords)
"print(tagged)"
