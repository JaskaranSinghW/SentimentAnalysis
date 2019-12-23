"""
Author: Jaskaran Singh 
Start Date: DEC 17 2019 
End Date: Dec __ 2019

Description: Take in Sentences from User Process it and output the sentiment

"""
import nltk
nltk.download('')

sentense = input("Enter Text to be Tested:")
print(sentense)

tokens = nltk.word_tokenize(sentense)

print(tokens)

tagged = nltk.pos_tag(tokens)
tagged[0:6]

print(tagged)
