#ML Asg 3
import csv
import math
import operator
import sys
import numpy
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer


wordList = []
vec = CountVectorizer()


# Load and read our data set
def loadDataset():
 	trainSet = []
 	train = open('train.csv')
 	trainData = list(csv.reader(train)) 	
 	for i in range(1, len(trainData)):
 		trainSet.append(trainData[i])

 	return trainSet



# iterates through dataSet and creates a list of all words
def wordListCreator(trainSet):
	for i in range(0,len(trainSet)):
		instanceList = trainSet[i]
		instanceWords = instanceList[2]
		instanceWords = re.sub(r'[^a-zA-Z ]', ' ', instanceWords)
		#tokenWords = nltk.word_tokenize(instanceWords)
		wordList.append(instanceWords)
	#print(wordList)
	return wordList


def filterNonAlphabetical(instanceWords):
	return [w for w in wordList if w.isalpha()]


# function to remove stopwords from our list of words
def preProcessingStopWords(tokenList):
	stopWords = set(stopwords.words('english'))
	filteredSentence = []
	for word in tokenList:
		if(word not in stopWords):
			filteredSentence.append(word)

	return filteredSentence

# creates a dictionary with the word and its occurence in the dataset
# key = word & value = count
def convertToDict(processedList):
	d = {}
	for word in processedList:
		d[word] = d.get(word, 0) + 1
	return d

# function to write our lists to a file.
def writeToFile(wordList, featureNames):

	w = open("wordList", 'w')
	f = open("featureList", 'w')

	for listitem in wordList:
		w.write('%s\n' % listitem)
	for listitem in featureNames:
		f.write('%s\n' % listitem)
	



def PreProcessing():
	#dataset in list form
	dataSet = loadDataset()

	#List of all the words in DataSet
	wordList = wordListCreator(dataSet)

	newList = filterNonAlphabetical(wordList)
	#for word in newList :
		#print(word)

	#Cleaned word list
	processedList = preProcessingStopWords(newList)

	#Converted words in dataset to dictionary with total appearances as value
	wordDictionary = convertToDict(processedList)

	#nltk feature extraction
	vectorizedMatrix = vec.fit_transform(wordList)
	#print(vectorizedMatrix)

	f = open("matrix" , 'w')
	for item in vectorizedMatrix.toarray():
		f.write('%s\n' % item)


	print(vectorizedMatrix.toarray())

	#vectorized list of features
	featureNames = vec.get_feature_names()


	writeToFile(wordList,featureNames)

PreProcessing()



















