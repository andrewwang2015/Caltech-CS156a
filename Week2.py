import numpy as np
import random
from datetime import datetime
MIN = -1.0
MAX = 1.0
NUMTRAININGPOINTS = 10
numIterations = 0
class Point:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def setScore(self, score):
        self.score = score;
class Line:
    def __init__(self, slope):
        self.slope = slope

def calculateG (p, weights):
    pointMatrix = np.matrix([0, p.x ,p.y])
    weightsMatrix = np.matrix([weights[0], weights[1], weights[2]])
    if (weightsMatrix * pointMatrix.transpose() > 0):
        return 1
    else:
        return -1

def returnRandomDouble(min1, max1):

    return random.uniform(min1, max1)

def updateWeightVector(weights, badPoint):
    global numIterations
    numIterations += 1
    weights[0] += badPoint.score;
    weights[1] += badPoint.x * badPoint.score;
    weights[2] += badPoint.y * badPoint.score;
    return


def runPLA(points, weights):
    for point in points:
        if (point.score != calculateG(point, weights)):
            updateWeightVector(weights, point);
            break;

def returnSlope(point1, point2):
    slope = ((point2.y - point1.y)/ (point2.x-point1.x))
    return slope

def findProbability(largeSet, weights):
    numDisagreeing = 0
    for point in largeSet:
        if (point.score != calculateG(point,weights)):
            numDisagreeing += 1;
    return (numDisagreeing / len(largeSet))

def isConverged(points, weights):
    for point in points:
        if (point.score != calculateG(point, weights)):
            return False
    return True

def setScore1 (currentPoint, targetLine, targetPoint):
    slope = targetLine.slope
    if ((currentPoint.y - targetPoint.y) > (slope * (currentPoint.x - targetPoint.x))):
        return 1
    else:
        return -1;
    
def main():
    global MIN
    global MAX
    global numIterations
    random.seed(datetime.now())
    samplePoints = []
    hellaPointsForProbability = []
    totalIterations = 0
    totalProbability = 0

    for j in range(1):
        numIterations = 0
        weightVector = [0.0, 0.0, 0.0]
        samplePoints.clear()
        hellaPointsForProbability.clear()
        startingPoint1 = Point(returnRandomDouble(MIN, MAX), returnRandomDouble(MIN, MAX))
        startingPoint2 = Point(returnRandomDouble(MIN,MAX), returnRandomDouble(MIN, MAX))
        targetLine = Line(returnSlope(startingPoint1, startingPoint2))
        for k in range(1):
            currentPoint = Point(returnRandomDouble(MIN, MAX), returnRandomDouble(MIN, MAX))
            currentPoint.setScore(setScore1(currentPoint, targetLine, startingPoint1))
            hellaPointsForProbability.append(currentPoint)
            
        
        for i in range(NUMTRAININGPOINTS):
            currentPoint = Point(returnRandomDouble(MIN, MAX), returnRandomDouble(MIN, MAX))
            currentPoint.setScore(setScore1(currentPoint, targetLine, startingPoint2))
            samplePoints.append(currentPoint)
        num = 0;
        
        while (isConverged(samplePoints, weightVector) == False):
            num += 1
            runPLA(samplePoints, weightVector)
            print (weightVector[0], weightVector[1], weightVector[2])
        #print (numIterations)
        #currentProbability = findProbability(hellaPointsForProbability, weightVector)
        #totalProbability += currentProbability;
        #totalIterations += numIterations
        
    #print (totalIterations / 1000)
    #print (totalProbability / 1000)

main()

                        
    