from sklearn import svm
import random
import numpy as np
import sys
from sklearn.cluster import KMeans
from numpy.linalg import *

class RBF:
    def __init__(self, gamma, numTrainingPoints):
        self.gamma = gamma
        self.inputs, self.outputs = self.generateData(numTrainingPoints)
        self.outSample, self.outSampleOuts = self.generateData(500)
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
    
    def calculateError(self, actualScores, predictedScores):
        assert(len(actualScores) == len(predictedScores))
        numWrong = 0
        for i in range(len(predictedScores)):
            if (int(predictedScores[i]) != int(actualScores[i])):
                numWrong +=1
        
        return numWrong / len(predictedScores)
    
    def runHardMargin(self, isInSample):
        if (isInSample):
            clf= svm.SVC(C=np.inf, kernel='rbf', gamma=self.gamma, coef0=1)
            clf.fit(self.inputs, self.outputs)
            predicted = clf.predict(self.inputs)
            inSampleError = self.calculateError(self.outputs, predicted)
            if (inSampleError != 0):
                return False
        else:
            clf = svm.SVC(C=np.inf, kernel='rbf', gamma=self.gamma,coef0=1)
            clf.fit(self.inputs, self.outputs)
            predicted = clf.predict(self.outSample)
            outSampleError = self.calculateError(self.outSampleOuts, predicted)
            return outSampleError
    
    def returnClusteredPrediction(self, input1, weights, centers):
        total = 0
        for i in range(len(centers)):
            squaredDiff = self.returnDifferenceSquared(input1, centers[i])
            total += weights[i+1] * np.exp(-1 * self.gamma * squaredDiff)
        total += weights[0]
        return np.sign(total)
    
    def runClustering(self, numClusters):
        km = KMeans(n_clusters=numClusters, n_init = 1)
        km.fit(self.inputs, self.outputs)
        #predicted = km.predict(self.outSample)
        clusters = km.cluster_centers_
        realClusters = []
        for i in clusters:
            realClusters.append(list(i))
        #print(realClusters[0])
        phi = []
        for i in range(self.numPoints):
            phi.append([1] * (numClusters + 1))
        
        for i in range(self.numPoints):
            for j in range(numClusters):
                squaredDiff = self.returnDifferenceSquared(self.inputs[i], realClusters[j])
                phi[i][j+1] = np.exp(-1 * self.gamma * squaredDiff)
        
        phiMatrix = np.matrix(phi)
        weights = pinv(phiMatrix) * np.matrix(self.outputs).transpose()
        predicted = []
        predictedIn = []
        for j in self.inputs:
            predictedIn.append(self.returnClusteredPrediction(j, weights, realClusters))
        #print (self.outSampleOuts)
        for i in self.outSample:
            predicted.append(self.returnClusteredPrediction(i, weights, realClusters))                        
        outSampleError = self.calculateError(predicted, self.outSampleOuts)
        inSampleError = self.calculateError(predictedIn, self.outputs)
        return inSampleError, outSampleError
    
    def returnDifferenceSquared(self, input1, input2):
        return (input1[0] - input2[0])**2 + (input1[1] - input2[1])**2
    
    def compare(self, numClusters):
        if (self.runHardMargin(False) < self.runClustering(numClusters)[1]):
            #print (self.runHardMargin(False), self.runClustering(numClusters))
            return True
        return False
	

        
if __name__ == '__main__':
    numRuns = 1000
    numWrong = 0
    gamma = 1.5
    numTrainingPoints = 100
    for i in range(numRuns):
        random.seed()
        run = RBF(gamma, numTrainingPoints)
        result = run.runHardMargin(True)
        if (result == False):
            numWrong += 1
    print("Ratio of times we get a dataset not separable by the RBF kernel: " + str(numWrong/numRuns))
    
    numRuns = 500
    numBetter = 0
    for j in range(numRuns):
        run1 = RBF(1.5, 100)
        result = run1.compare(9)
        if (result):
            numBetter += 1
    print ("Ratio of times the kernel form beats the regular form(k=9) in E_out: " + str(numBetter/ numRuns))
    
    numBetter = 0
    for j in range(numRuns):
        run1 = RBF(1.5, 100)
        result = run1.compare(12)
        if (result):
            numBetter += 1
    print ("Ratio of times the kernel form beats the regular form(k=12) in E_out: " + str(numBetter/ numRuns))   
    
    eInChange = 0
    eOutChange = 0
    numRuns = 100
    for i in range(numRuns):
        run2 =  RBF(1.5, 100)
        k_9 = run2.runClustering(9)
        k_12 = run2.runClustering(12)
        if (k_12[0] < k_9[0]):
            eInChange += 1 
        if (k_12[1] < k_9[1]):
            eOutChange += 1
    print ("Number of times E_in decrease from k = 9 to k = 12: " + str(eInChange) + ". Number of times E_out decrease from k = 9 to k = 12: " + str(eOutChange) + ".")    
   
   
    eInChange = 0
    eOutChange = 0
    numRuns = 100
    for i in range(numRuns):
        run3 =  RBF(1.5, 100)
        run4 = RBF(2.0, 100)
        gamma1 = run3.runClustering(9)
        gamma2 = run4.runClustering(9)
        if (gamma2[0] < gamma1[0]):
            eInChange += 1 
        if (gamma2[1] < gamma1[1]):
            eOutChange += 1
    print ("Number of times E_in decrease from g = 1.5 to g = 2: " + str(eInChange) + ". Number of times E_out decrease from g = 1.5 to g = 2: " + str(eOutChange) + ".")    
    
    numRuns = 500
    numAchieved = 0
    for i in range(numRuns):
        run5 = RBF(1.5, 100)
        result = run5.runClustering(9)
        if (result[0] - 0 < 0.00001):
            numAchieved += 1
    print ("Percentage of time that regular RBF (k=9, gamma = 1.5) achieves E_in = 0: " + str(numAchieved/numRuns))
    

            
        
        