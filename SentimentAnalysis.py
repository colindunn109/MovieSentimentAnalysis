#ML Asg 3
import csv
import math
import operator
import sys
import numpy
import nltk

nltk.download('punkt')


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


def wordDictionary(trainSet):
	for i in range(0,2):
		instanceList = trainSet[i]
		instanceWords = instanceList[2]
		tokenWords = nltk.word_tokenize(instanceWords)
		print(instanceWords)
		print(tokenWords)

		
		

print("hello")
dataSet = loadDataset()

wordDictionary(dataSet)
