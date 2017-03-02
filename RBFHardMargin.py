from sklearn import svm
import random
import numpy as np
import sys
from sklearn.cluster import KMeans

class RBF:
    def __init__(self, gamma, numTrainingPoints):
        self.gamma = gamma
        self.inputs, self.outputs = self.generateData(numTrainingPoints)
        self.outSample, self.outSampleOuts = self.generateData(1000)
        self.numPoints = numTrainingPoints

    def generateData(self, numPoints):
        inputs = []
        targetValues = []
        for i in range(numPoints):
            x = random.random() * 2 - 1
            y = random.random() * 2 - 1
            inputs.append([x,y])
            score = self.returnTargetValue(x, y)
            targetValues.append(score)
        return inputs, targetValues   
    
    def returnTargetValue(self, x, x2):
        return np.sign(x2 - x + 0.25 * (np.sin(np.pi * x)))
    
    def calculateError(self, predictedScores, actualScores):
        numWrong = 0
        for i in range(len(predictedScores)):
            if predictedScores[i] != actualScores[i]:
                numWrong +=1
        return numWrong / len(predictedScores)
        
    def runHardMargin(self, isInSample):
        if (isInSample):          
            clf = svm.SVC(C=np.inf, kernel='rbf', gamma=self.gamma, coef0=1)
            clf.fit(self.inputs, self.outputs)
            predicted = clf.predict(self.inputs)
            inSampleError = self.calculateError(self.outputs, predicted)
            if (inSampleError != 0):
                return False

	
 
    def runClustering(self, numClusters):
        km = KMeans(k=numClusters)
        km.fit(self.inputs, self.outputs)
        predicted = km.predict(self.outSample)
        outSampleError = self.calculateError(self.outSampleOuts, predicted)
        return outSampleError
    
    
    def compareClusteringAndRegular(self, numClusters):
	if (self.runHardMargin(False) < self.runClustering(numClusters)):
	    return True
	return False
        
if __name__ == '__main__':
    #numRuns = 1000
    #numWrong = 0
    #for i in range(numRuns):
        #random.seed()
        #run = RBF(1.5, 100)
        #result = run.runHardMargin(True)
        #if result == False:
            #numWrong += 1
    #print (numWrong/numRuns)
    
    #numRuns = 1000
    #numBetter = 0
    
    #for i in range(numRuns):
	#run = RBF(1.5, 100)
	#result = run.compareClusteringAndRegular(9)
	#if (result):
	    #numBetter += 1
    #print (numBetter / numRuns)
            
        
        