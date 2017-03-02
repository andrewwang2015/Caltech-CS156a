from sklearn import svm
import random
import numpy as np
import sys

    
class PLA_SVM:
    def __init__(self, numPoints):
        self.numSV = 0
        self.outOfSampleError = 0
        self.SVMError = 0
        self.misclassifiedPoints = []
        self.numPoints = numPoints
        self.numIterations = 0
        self.weights = [0, 0, 0]
        self.initialPoint = []
        self.lineSlope = 0
        self.generateTargetFunction()
        self.inSample = self.generateRandomPointsIn(self.numPoints)
        self.outSample = self.generateRandomPoints(1000)
    
    def generateRandomPoints(self, numPoints):
        points = []
        for i in range(numPoints):
            x = random.random() * 2 - 1
            y = random.random() * 2 - 1
            score = self.returnScore(x, y)
            point = [x, y, score]
            points.append(point)
        return points
    
    def generateRandomPointsIn(self, numPoints):
        while (True):
            points = []
            scores = []
            for i in range(numPoints):
                x = random.random() * 2 - 1
                y = random.random() * 2 - 1
                score = self.returnScore(x, y)
                scores.append(score)
                point = [x, y, score]
                points.append(point)
            if (scores.count(scores[0]) != len(scores)):
                break
        return points    
    
    def calculateError(self):
        numWrong = 0
        for point in self.outSample:
            if (self.isPointMisclassified(point)):
                numWrong += 1
        return numWrong / (len(self.outSample))
                
    def generateTargetFunction(self):
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1
        x1 = random.random() * 2 - 1
        y1 = random.random() * 2 - 1
        slope = (y - y1) / (x - x1)
        self.initialPoint = [x, y]
        self.lineSlope = slope
        
    def returnScore(self, x, y):
        if (y - self.initialPoint[1] > self.lineSlope * (x - self.initialPoint[0])):
            return 1
        else:
            return -1
    
    def continuePLA(self):
        flag = 0
        self.misclassifiedPoints.clear()
        for point in self.inSample:
            if (self.isPointMisclassified(point)):
                self.misclassifiedPoints.append(point)
                flag = 1
        if (flag != 0):
            return True
        return False
    
    def isPointMisclassified(self, point):
        tempPoint = [point[0], point[1], 1]
        dotProduct = np.dot(self.weights, tempPoint)
        if (np.sign(dotProduct) == point[2]):
            return False
        return True
    
    def adjustWeights(self, point):
        self.numIterations += 1
        self.weights[0] += point[2] * point[0]
        self.weights[1] += point[2] * point[1]
        self.weights[2] += point[2]
    
    def runPLA(self):
        while (self.continuePLA() == True):
            rand = random.randint(0, len(self.misclassifiedPoints) - 1)
            randPoint = self.misclassifiedPoints[rand]
            self.adjustWeights(randPoint)
        self.outOfSampleError = self.calculateError()
        
    def calculateSVMError(self, predictedScores, actualScores):
        numWrong = 0
        for i in range(len(predictedScores)):
            if predictedScores[i] != actualScores[i]:
                numWrong +=1
        return numWrong / len(predictedScores)
        
    def runSVM(self):

        svmPoints = []
        svmScores = []
        svmPointsOut = []
        svmScoresOut = []
        
        for point in self.inSample:
            currentPoint = [point[0], point[1]]
            svmPoints.append(currentPoint)
            svmScores.append(point[2])
            
        clf = svm.SVC(kernel='linear')
        clf.C = sys.maxsize            
        clf.fit(svmPoints, svmScores)
        self.numSV = len(clf.support_vectors_)
       # print (self.numSV)
        for point in self.outSample:
            currentPoint = [point[0], point[1]]
            svmPointsOut.append(currentPoint)
            svmScoresOut.append(point[2])  
            
        svmPredictedScores = clf.predict(svmPointsOut)
        #print (len(svmPredictedScores), len(svmScoresOut))
        self.SVMError = self.calculateSVMError(svmPredictedScores, svmScoresOut) 
            
if __name__ == '__main__':
    numBetter = 0
    numSupportVectors = 0
    for i in range(1000):
        random.seed()
        p = PLA_SVM(100)
        p.runPLA()
        p.runSVM()
        if (p.SVMError < p.outOfSampleError):
            numBetter += 1
        numSupportVectors += p.numSV
    print (numBetter / 1000)
    print (numSupportVectors / 1000)
    