from __future__ import print_function, division
from builtins import range

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

from collections import Counter
import random

# read the dataset
data = pd.read_csv('./data/dataset.csv', header=0)
data.columns = ['text', 'dataset', 'fyhao', 'twinword', 'synmato']
# print(data) # test purpose
manual = np.array(data.dataset)
api_1 = np.array(data.fyhao)
api_2 = np.array(data.twinword)
api_3 = np.array(data.synmato)

class GloveVectorizer:
  def __init__(self):
    # load in pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    embedding = []
    idx2word = []
    with open('./glove.6B/glove.6B.50d.txt') as f:
      # open space-separated text vector dictionary:
      # word vec[0] vec[1] vec[2] ...
      for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
    print('Found %s word vectors.' % len(word2vec))

    # initialization
    self.word2vec = word2vec
    self.embedding = np.array(embedding)
    self.word2idx = {v:k for k,v in enumerate(idx2word)}
    self.V, self.D = self.embedding.shape

  def fit(self, data):
    pass

  def transform(self, data):
    # declare a new array with the same length and dimensions of the dataset
    Xmean = np.zeros((len(data), self.D))
    Xsum = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.lower().split()
      vecs = []
      for word in tokens:
        if word in self.word2vec:
          vec = self.word2vec[word]
          vecs.append(vec)
      if len(vecs) > 0:
        vecs = np.array(vecs)
        # print(vecs) # Test vecs
        Xmean[n] = vecs.mean(axis=0) # get mean along the column to transform the data
        Xsum[n] = vecs.sum(axis=0)
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return Xsum

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)


# vectorize the dataset
vectorizer = GloveVectorizer()
Xdata = vectorizer.fit_transform(data.text)

# define the number of the clusters
k = 3

# Create a k-means model and fit it to the data
km = KMeans(n_clusters=k)
km.fit(Xdata)

# Predict the clusters for each document
y_pred = km.predict(Xdata)

# Print the cluster assignments
print(y_pred)

def CompareAccuracy(api):
  randlist1 = random.sample(range(0, elem[0]), round(elem[0]*0.1))
  randlist2 = random.sample(range(0, elem[1]), round(elem[1]*0.1))
  randlist3 = random.sample(range(0, elem[2]), round(elem[2]*0.1))
  sample = []
  arr = []
  for num in randlist1:
    key = elem_0[num]
    sample.append(api[key])
    arr.append(manual[key])
  for num in randlist2:
    key = elem_1[num]
    sample.append(api[key])
    arr.append(manual[key])
  for num in randlist3:
    key = elem_2[num]
    sample.append(api[key])
    arr.append(manual[key])
  accurate = accuracy(sample, arr)
  return accurate

# compare two arrays to get accuracy
def accuracy(sample, arr):
  sample = np.array(sample)
  arr = np.array(arr)
  comm = np.where(sample == arr)[0]
  accurate = len(comm)/len(arr)
  return accurate

# calculate the number of elements in each cluster
elem = Counter(y_pred)
print("\nk-clusters count:\t", elem)
elem_0 = np.where(y_pred == 0)[0]
elem_1 = np.where(y_pred == 1)[0]
elem_2 = np.where(y_pred == 2)[0]
def ClusterAccuracy():
  print("\nk-Clustering Accuracy:\nAPI 1\tAPI 2\tAPI 3")
  print("{0:.2%}".format(CompareAccuracy(api_1)), "\t", "{0:.2%}".format(CompareAccuracy(api_2)), " ", "{0:.2%}".format(CompareAccuracy(api_3)))


# calculate total accuracy of 3 APIs
def PopulationAccuracy():
  accuracy1 = accuracy(api_1, manual)
  accuracy2 = accuracy(api_2, manual)
  accuracy3 = accuracy(api_3, manual)
  print("\nTotal Population Accuracy: \nAPI 1\tAPI 2\tAPI 3")
  print("{0:.2%}".format(accuracy1), "\t", "{0:.2%}".format(accuracy2), " ", "{0:.2%}".format(accuracy3))

# random generate 200 samples and calculate the accuracy
def RandomSampleAccuracy():
  randomIndex = random.sample(range(0,2000), 200)
  randomIndex.sort()
  sample = []
  sample_api_1 = []
  sample_api_2 = []
  sample_api_3 = []
  for num in randomIndex:
    sample.append(manual[num])
    sample_api_1.append(api_1[num])
    sample_api_2.append(api_2[num])
    sample_api_3.append(api_3[num])
  accuracy1 = accuracy(sample_api_1, sample)
  accuracy2 = accuracy(sample_api_2, sample)
  accuracy3 = accuracy(sample_api_3, sample)
  print("\nRandom Pick Sample Accuracy: \nAPI 1\tAPI 2\tAPI 3")
  print("{0:.2%}".format(accuracy1), "\t", "{0:.2%}".format(accuracy2), " ", "{0:.2%}".format(accuracy3))

RandomSampleAccuracy()
PopulationAccuracy()
ClusterAccuracy()