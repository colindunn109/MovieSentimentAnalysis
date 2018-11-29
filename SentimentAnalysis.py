#ML Asg 3
import csv
import math
import operator
import sys
import numpy
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')

wordList = []
# Load and read our data set

def loadDataset():
 	trainSet = []
 	train = open('train.csv')
 	trainData = list(csv.reader(train))

 	print(trainData[0])
 	

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
	print(wordList)
	return wordList

def preProcessingStopWords(tokenList):
	stopWords = set(stopwords.words('english'))
	filteredSentence = []
	for word in tokenList:
		if(word not in stopWords):
			filteredSentence.append(word)

	return filteredSentence



		
		

dataSet = loadDataset()

temp = wordListCreator(dataSet)
temp2 = preProcessingStopWords(temp)
print(temp2)
