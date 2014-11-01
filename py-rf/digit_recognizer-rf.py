import csv
import numpy as np
from numpy import genfromtxt, savetxt
from sklearn.ensemble import RandomForestClassifier
 

#Specify training data, testing data, and labeled data file locations
ifile  = open('../Data/train.csv', "r")
tfile = open('../Data/test.csv',"r")

data = genfromtxt(ifile,delimiter=',',dtype='f8')[1:]
labels = [x[0] for x in data]
train =[x[1:] for x in data]
test = genfromtxt(tfile,delimiter=',',dtype='f8')[1:]

#Train data
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train,labels)

#Classify
classified = rf.predict(test)

#Write
savetxt('../Data/submission-rf-py.csv',classified,delimiter=',',fmt='%f')