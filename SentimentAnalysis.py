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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



wordList = []
labelList = []
vec = CountVectorizer()
vectorizedMatrix = []


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
	returnList = []
	for phrase in tokenList:
		words = phrase.split(' ')
		tempString = ""
		for word in words:
			if(word not in stopWords):
				tempString += word
				tempString += " "
		returnList.append(tempString)


	return returnList

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

# funciton to create a list of labels given the DataSet

def getLabelList(trainSet):
	for i in range(0, len(trainSet)):
		trainInstance = trainSet[i]
		labelList.append(trainInstance[3])
	return labelList



def PreProcessing():
	#dataset in list form
	dataSet = loadDataset()

	#List of all the phrases in DataSet
	wordList = wordListCreator(dataSet)

	#List of all the Labels in the Data Set
	labelList = getLabelList(dataSet)

	#Cleaned word list
	processedList = preProcessingStopWords(wordList)
	#for i in processedList:
		#print(i)

	#Converted words in dataset to dictionary with total appearances as value
	wordDictionary = convertToDict(processedList)

	#nltk feature extraction
	vectorizedMatrix = vec.fit_transform(wordList)


	#vectorized list of features
	featureNames = vec.get_feature_names()

	return vectorizedMatrix, labelList


	#writeToFile(wordList,featureNames)


dataMatrix, labelList = PreProcessing()

def trainingData(vectorizedMatrix, labelList):

	X = vectorizedMatrix.toarray()
	y = labelList

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

	classifier = GaussianNB()
	classifier.fit(X_train, y_train)

	prediction = classifier.predict(X_test)

	accuracy = accuracy_score(y_test, prediction)
	print(accuracy)


trainingData(dataMatrix, labelList)


















