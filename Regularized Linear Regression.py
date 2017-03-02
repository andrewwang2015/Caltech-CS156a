import numpy as np
import random
from numpy.linalg import inv

class RegularizedLinearRegression:
    def __init__(self, training, testing, transformed):
        self.trainingSet, self.trainingSetDigits = self.loadData(training)
        self.testingSet, self.testingSetDigits = self.loadData(testing)
        self.regularizedWeights = []
        self.transformedTrainingInputs = []
        self.transformedTestingInputs = []
        self.toTransform = transformed
        self.versusTrainingInputs = []
        self.versusTestingInputs = []

    def loadData(self, fileName):
        f = open(fileName, 'r')
        digits = []
        data = []
        for line in f:
            digit, intensity, symmetry = line.split()
            data.append([1, float(intensity), float(symmetry)])
            digits.append([int(float(digit))])
        return data, digits
    
    def returnVersusPredictedY(self, num1, num2, isTraining):
        x = []
        if isTraining:
            if (self.toTransform):
                for i in range(len(self.trainingSetDigits)):
                    if (self.trainingSetDigits[i][0] == num1 or self.trainingSetDigits[i][0] == num2):
                        x.append(self.transformedTrainingInputs[i])
            else:
                for i in range(len(self.trainingSetDigits)):
                    if (self.trainingSetDigits[i][0] == num1 or self.trainingSetDigits[i][0] == num2):
                        x.append(self.trainingSet[i])
        else:
            if (self.toTransform):
                for i in range(len(self.testingSetDigits)):
                    if (self.testingSetDigits[i][0] == num1 or self.testingSetDigits[i][0] == num2):
                        x.append(self.transformedTestingInputs[i])
            else:
                for i in range(len(self.testingSetDigits)):
                    if (self.testingSetDigits[i][0] == num1 or self.testingSetDigits[i][0] == num2):
                        x.append(self.testingSet[i])
        
        actualResults = []
        for i in x:
            actualResults.append(np.sign(np.dot(list(np.squeeze(self.regularizedWeights)), i)))
        return actualResults        
                
    def returnVersusAllYPredictedY(self, isTraining):
        if isTraining:
            if (self.toTransform):
                x = self.transformedTrainingInputs
            else:
                x = self.trainingSet
        else:
            if (self.toTransform):
                x = self.transformedTestingInputs
            else:
                x = self.testingSet
        actualResults = []
        for i in x:
            actualResults.append(np.sign(np.dot(list(np.squeeze(self.regularizedWeights)), i)))
        return actualResults
    
    def error(self, num, isTraining):
        return self.calculateError(self.returnVersusAllY(num, isTraining), self.returnVersusAllYPredictedY(isTraining))
    
    def errorVersus(self, num1, num2, isTraining):
        return self.calculateError(self.returnVersusY(num1, num2, isTraining), self.returnVersusPredictedY(num1, num2, isTraining))
    
    def calculateError(self, predictedScores, actualScores):
        numWrong = 0
        for i in range(len(predictedScores)):
            if predictedScores[i] != actualScores[i]:
                numWrong +=1
        return numWrong / len(predictedScores)
    
    
    def setUpTransformedPoints(self):
   
        temp = []
        for i in self.trainingSet:
            temp.append([1, i[1], i[2], i[1] * i[2], i[1] * i[1], i[2] * i[2]])
        self.transformedTrainingInputs = temp
        
        temp1 = []
        for i in self.testingSet:
            temp1.append([1, i[1], i[2], i[1] * i[2], i[1] * i[1], i[2] * i[2]])
        self.transformedTestingInputs = temp1      
    
    def setUpVersusPoints(self, num1, num2):
        x = []
        for i in range(len(self.trainingSetDigits)):
            if (self.trainingSetDigits[i][0] == num1 or self.trainingSetDigits[i][0] == num2):
                x.append(self.transformedTrainingInputs[i])
        self.versusTrainingInputs = x
        
        y = []
        for i in range(len(self.testingSetDigits)):
            if (self.testingSetDigits[i][0] == num1 or self.testingSetDigits[i][0] == num2):
                y.append(self.transformedTestingInputs[i])
        self.versusTestingInputs = y  
        
    def runRegression(self, multiplier, versusAll, isVersusAll, num1, num2):
        self.setUpTransformedPoints()
        self.setUpVersusPoints(num1, num2)
        if (self.toTransform):
            if (isVersusAll == False):
                newTrainingSet = self.versusTrainingInputs
            else:
                newTrainingSet = self.transformedTrainingInputs
        else:
            newTrainingSet = self.trainingSet
        
        if (isVersusAll):
            y = np.matrix(self.returnVersusAllY(versusAll, True))
        else:
            y = np.matrix(self.returnVersusY(num1, num2, True))
        Z = np.matrix(newTrainingSet)
        sizeOfIdentity = Z.shape[1]
        w_reg = inv(Z.transpose() * Z + multiplier * np.identity(sizeOfIdentity)) * Z.transpose() * y
        self.regularizedWeights = w_reg      
        
    def returnVersusAllY(self, i, isTraining):
        if isTraining:
            x = self.trainingSetDigits
        else:
            x = self.testingSetDigits
        actualResults = []
        for digit in x:
            if (digit[0] == i):
                actualResults.append([1])
            else:
                actualResults.append([-1])
        return actualResults
    
    def returnVersusY(self, num1, num2, isTraining):
        if isTraining:
            x = self.trainingSetDigits
        else:
            x = self.testingSetDigits
        actualResults = []
        for digit in x:
            if (digit[0] == num1):
                actualResults.append([1])
            elif(digit[0] == num2):
                actualResults.append([-1])
        return actualResults    

        
if __name__ == '__main__':
    reg = RegularizedLinearRegression('features.train.txt', 'features.test.txt', False)
    minError = 999
    minProblem7 = 0
    k = 1
    for i in range(5,10):
        reg.runRegression(k, i, True, 0, 0)
        error = reg.error(i, True)
        if error < minError:
            minError = error
            minProblem7 = i
    print ("Answer for problem 7: " + str(minProblem7) + " versus all with an error of " + str(minError))
    
    reg1 = RegularizedLinearRegression('features.train.txt', 'features.test.txt', True)
    minError = 999
    minProblem8 = 0
    for i in range(0,5):
        reg1.runRegression(k, i, True, 0, 0)
        error = reg1.error(i, False)
        if error < minError:
            minError = error
            minProblem8 = i
    print ("Answer for problem 8: " + str(minProblem8) + " versus all with an error of " + str(minError))    
    
    reg2 = RegularizedLinearRegression('features.train.txt', 'features.test.txt', False)
    reg2.runRegression(k,5, True, 0, 0)
    error5_noTransform = reg2.error(5, False)
    
    reg3 = RegularizedLinearRegression('features.train.txt', 'features.test.txt', True)
    reg3.runRegression(k,5, True, 0, 0)
    error5_transform = reg3.error(5, False)   
    
    print ("The error without transformation for '5 versus all' is " + str(error5_noTransform) + " and with transform, is " + str(error5_transform))
    
    reg4 = RegularizedLinearRegression('features.train.txt', 'features.test.txt', True)
    reg4.runRegression(1, 'daf', False, 1, 5)
    print(reg4.errorVersus(1,5, True))
    
    reg5 = RegularizedLinearRegression('features.train.txt', 'features.test.txt', True)
    reg5.runRegression(0.01, 'daf', False, 1, 5)
    print(reg5.errorVersus(1,5, True))    