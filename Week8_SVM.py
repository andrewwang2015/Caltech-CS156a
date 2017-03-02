from sklearn import svm
from random import shuffle
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from statistics import mode
import random

class SVM_Set8:
    def __init__(self, training, testing):
        self.numSV = 0
        self.outOfSampleError = 0
        self.inSampleError = 0
        self.numIterations = 0
        self.trainingSet, self.trainingSetDigits = self.loadData(training)
        self.testingSet, self.testingSetDigits = self.loadData(testing) 
    
    def loadData(self, fileName):
        f = open(fileName, 'r')
        digits = []
        data = []
        for line in f:
            digit, intensity, symmetry = line.split()
            data.append([float(intensity), float(symmetry)])
            digits.append(int(float(digit)))
        return data, digits
    
    
    def calculateSVMError(self, predictedScores, actualScores):
        numWrong = 0
        for i in range(len(predictedScores)):
            if predictedScores[i] != actualScores[i]:
                numWrong +=1
        return numWrong / len(predictedScores)
    
    def returnNumWrong(self, predictedScores, actualScores):
        numWrong = 0
        for i in range(len(predictedScores)):
            if predictedScores[i] != actualScores[i]:
                numWrong +=1
        return numWrong 
    
    
    def versusAllForLeastError(self):
        possibleChoices = [1, 3, 5, 7, 9]
        actualResults = []
        predictedResults = []
        minError = 999
        minChoice = 0
        numSupportVecs = 0
        for i in possibleChoices:

            actualResults = []
            for digit in self.trainingSetDigits:
                if (digit == i):
                    actualResults.append(1)
                else:
                    actualResults.append(-1)
            clf = svm.SVC(degree=2, kernel='poly')
            clf.C = 0.01
            
            clf.fit(self.trainingSet, actualResults)
            predictedResults = clf.predict(self.trainingSet)
            error = self.calculateSVMError(predictedResults, actualResults)
            if error < minError:
                minError = error
                minChoice = i
                numSupportVecs = len(clf.support_vectors_)
        return minChoice, minError, numSupportVecs
    
    
    def versusAllForGreatestError(self):
        possibleChoices = [0, 2, 4, 6, 8]
        actualResults = []
        predictedResults = []
        maxError = -999
        maxChoice = 0
        numSupportVecs = 0
        for i in possibleChoices:

            actualResults = []
            for digit in self.trainingSetDigits:
                if (digit == i):
                    actualResults.append(1)
                else:
                    actualResults.append(-1)
            clf = svm.SVC(degree=2, kernel='poly')
            clf.C = 0.01
            clf.fit(self.trainingSet, actualResults)
            predictedResults = clf.predict(self.trainingSet)
            error = self.calculateSVMError(predictedResults, actualResults)
            if error > maxError:
                maxError = error
                maxChoice = i
                numSupportVecs = len(clf.support_vectors_)
        return maxChoice, maxError, numSupportVecs
    
    def versus(self, num1, num2, c_svm):
        numSupportVecs = 0
        actualResults = []
        actualData = []
        for i in range(len(self.trainingSetDigits)):
            if (self.trainingSetDigits[i] == num1):
                actualResults.append(1)
                actualData.append(self.trainingSet[i])
            elif (self.trainingSetDigits[i] == num2):
                actualResults.append(-1)
                actualData.append(self.trainingSet[i])
        clf = svm.SVC(degree=2, kernel='poly')
        clf.C = c_svm
        clf.fit(actualData, actualResults)
        predictedResults = clf.predict(actualData)
        error = self.calculateSVMError(predictedResults, actualResults)
        numSupportVecs = len(clf.support_vectors_)
        return error, numSupportVecs
    
    def runCV(self, num1, num2, possibleC):
        actualResults = []
        actualData = []
        for i in range(len(self.trainingSetDigits)):
            if (self.trainingSetDigits[i] == num1):
                actualResults.append(1)
                actualData.append(self.trainingSet[i])
            elif (self.trainingSetDigits[i] == num2):
                actualResults.append(-1)
                actualData.append(self.trainingSet[i])   
        combined = list(zip(actualData,actualResults))
        shuffle(combined)
        actualData[:], actualResults[:] = zip(*combined)
        tempTestingSet = actualData[-157:]
        tempTestingSetScores = actualResults[-157:]
        tempTrainingSet = actualData[:-157]
        tempTrainingSetScores = actualResults[:-157]
        minError = 999
        minC = 0
        
        for c in possibleC:
            clf = svm.SVC(degree=2, kernel='poly', coef0=1.0)
            clf.C = c
            clf.fit(tempTrainingSet, tempTrainingSetScores)
            predictedScores = clf.predict(tempTestingSet)
            error = self.calculateSVMError(predictedScores, tempTestingSetScores)
            if (error < minError):
                minError = error
                minC = c
        return minC, minError
        
    def runRBF(self, num1, num2, possibleC):
        actualResults = []
        actualData = []
        for i in range(len(self.trainingSetDigits)):
            if (self.trainingSetDigits[i] == num1):
                actualResults.append(1)
                actualData.append(self.trainingSet[i])
            elif (self.trainingSetDigits[i] == num2):
                actualResults.append(-1)
                actualData.append(self.trainingSet[i])          
                minError = 999
                minC = 0
        actualResultsOut = []
        actualDataOut = []
        for i in range(len(self.testingSetDigits)):
            if (self.testingSetDigits[i] == num1):
                actualResultsOut.append(1)
                actualDataOut.append(self.testingSet[i])
            elif (self.testingSetDigits[i] == num2):
                actualResultsOut.append(-1)
                actualDataOut.append(self.testingSet[i])          
        minError = 999
        minC = 0                
        for c in possibleC:
            clf = svm.SVC(kernel = 'rbf')
            clf.C = c
            clf.fit(actualData, actualResults)
            predictedScores = clf.predict(actualDataOut)
            error = self.calculateSVMError(predictedScores, actualResultsOut)
            if (error < minError):
                minError = error
                minC = c
        return minC, minError
            
if __name__ == '__main__':
    svm1 = SVM_Set8('features.train.txt', 'features.test.txt')
    problem2Choice, problem2Error, problem2supportvectors = svm1.versusAllForGreatestError()
    problem3Choice, problem3Error, problem3supportvectors = svm1.versusAllForLeastError()
    print (problem2Choice, problem2Error, problem2supportvectors)
    print (problem3Choice, problem3Error, problem3supportvectors)
    minVersusError = 999
    minC = 0
    possibleC = [0.001, 0.01, 0.1, 1]
    for i in possibleC:
        currentError, numberSupportVectorsVersus  = svm1.versus(1, 5, i)
        if (currentError <= minVersusError):
            minVersusError = currentError
            minC = i
    print (minC, minVersusError)
    error, numberSupportVectorsVersus  = svm1.versus(1, 5, 0.001)
    print (numberSupportVectorsVersus)
    random.seed()
    possibleC_problem7 = [0.0001, 0.001, 0.01, 0.1, 1]
    allC = []
    allError = []
    for k in range(100):
        c, error = (svm1.runCV(1,5, possibleC_problem7))
        allC.append(c)
        allError.append(error)
    print (mode(allC))
    print (sum(allError)/len(allError))
    allC = []
    possibleC_problem8 = [0.01, 1, 100, 10000, 1000000]
    for k in range(1):
        c, error = svm1.runRBF(1,5, possibleC_problem8)
        print (c, error)