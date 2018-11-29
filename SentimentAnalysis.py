#ML Asg 3
import csv
import math
import operator
import sys
import numpy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer


wordList = []
vec = DictVectorizer()


# Load and read our data set
def loadDataset():
 	trainSet = []
 	train = open('train.csv')
 	trainData = list(csv.reader(train))

 	print(trainData[0])
 	

 	for i in range(1, len(trainData)):
 		trainSet.append(trainData[i])

 	return trainSet



# iterates through dataSet and creates a list of all words
def wordListCreator(trainSet):
	for i in range(0,len(trainSet)):
		instanceList = trainSet[i]
		instanceWords = instanceList[2]
		tokenWords = nltk.word_tokenize(instanceWords)
		wordList.extend(tokenWords)
	#print(wordList)
	return wordList


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
	



#dataset in list form
dataSet = loadDataset()

#List of all the words in DataSet
temp = wordListCreator(dataSet)

#Cleaned temp
temp2 = preProcessingStopWords(temp)

#Converted words in dataset to dictionary with total appearances as value
ourDict = convertToDict(temp2)

#nltk feature extraction
vec.fit_transform(ourDict).toarray()

#vectorized list of features
featureNames = vec.get_feature_names()

#writeToFile(temp2, featureNames)




















