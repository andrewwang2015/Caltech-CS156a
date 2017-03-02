from sklearn import svm
from random import shuffle
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from statistics import mode
import random
import sys

X = [[1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0]]
Y = [-1, -1, -1, 1, 1, 1, 1]

clf = svm.SVC(degree=2, kernel='poly', coef0 = 1)
clf.C = sys.maxsize

clf.fit(X,Y)

print(len(clf.support_vectors_))

