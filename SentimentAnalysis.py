#ML Asg 3
import csv
import math
import operator
import sys
import numpy
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from sklearn.feature_extraction import DictVectorizer


wordList = []
vec = DictVectorizer()
# Load and read our data set

def loadDataset():
 	trainSet = []
 	train = open('train.csv')
 	trainData = list(csv.reader(train)) 	

 	for i in range(1, len(trainData)):
 		trainSet.append(trainData[i])

 	return trainSet


# create a dictionary such that key = word
# and value is the amount of times the word is found


def wordListCreator(trainSet):
	for i in range(0,len(trainSet)):
		instanceList = trainSet[i]
		instanceWords = instanceList[2]
		tokenWords = nltk.word_tokenize(instanceWords)
		wordList.extend(tokenWords)
	return wordList

def preProcessingStopWords(tokenList):
	stopWords = set(stopwords.words('english'))
	filteredSentence = []
	for word in tokenList:
		if(word not in stopWords):
			filteredSentence.append(word)

	return filteredSentence

def convertToDict(processedList):
	d = {}
	for word in processedList:
		d[word] = d.get(word, 0) + 1
	return d


def getSentiment(words):
	sentimentAnalyzer = SentimentAnalyzer()
	unigramFeats = sentimentAnalyzer.unigram_word_feats(words)
	#what do we do with the unigrams?
		


dataSet = loadDataset()

temp = wordListCreator(dataSet)
temp2 = preProcessingStopWords(temp)
temp3 = convertToDict(temp2)
temp4 = getSentiment(temp3)
ourDict = convertToDict(temp2)
vec.fit_transform(ourDict).toarray()
print(vec.get_feature_names())



