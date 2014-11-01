import csv
import numpy as np
from numpy import genfromtxt, savetxt
from sklearn import RandomForestClassifier
 

#Specify training data, testing data, and labeled data file locations
ifile  = open('../Data/train.csv', "r")
tfile = open('../Data/test.csv',"r")

data = genfromtxt(ifile,delimiter=',',dtype='f8')[1:]
labels = [x[0] for x in dataset]
train =[x[1] for x in dataset]
test = genfromtxt(tfile,delimiter=',',dtype='f8')[1:]

#Preprocessing -- Scale data
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
          	  [ 0.,  1., -1.]])

X_scaled = preprocessing.scale(X)

print(X_scaled)

#Train data
rf = RandomForestClassifierm(n_estimators=100)
rf.fit(train,target)

#Classify
classified = rf.predict(test)

#Write
savetxt('../Data/submission-rf-py.csv',classified,delimiter=',',fmt='%f')