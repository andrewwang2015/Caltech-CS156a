#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <armadillo>
#include <algorithm>
using namespace std;
using namespace arma;
#define MIN -1.0
#define MAX 1.0
#define NUMTRAININGPOINTS  10

int numIterations = 0;

typedef struct _point
{
    double x;
    double y;
    int score;
} point;

typedef struct _line
{
    double slope;
} line;

int calculateG (point p, double weights[])
{
    double innerProd = weights[0] + weights[1] * p.x + weights[2] * p.y;
    if (innerProd > 0)
        return 1;
    else
        return -1;
}

double returnRandomDouble(double min, double max)
{
    double randomNum = (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
    return randomNum;
}

void updateWeightVector (double weights[], point badPoint)
{
    numIterations++;
    weights[0] += badPoint.score;
    weights[1] += badPoint.x * badPoint.score;
    weights[2] += badPoint.y * badPoint.score;
}

void runPLA(vector<point> points, double weights[])
{
    random_shuffle(points.begin(), points.end());
    for (point p : points)
    {
        if (p.score != calculateG(p, weights))
        {
            updateWeightVector(weights, p);
            break;
        }
    }
}

line returnTargetFunction(point point1, point point2)
{
    line targetLine;
    double targetSlope = ((point2.y) - (point1.y)) / ((point2.x) - (point1.x));
    targetLine.slope = targetSlope;
    return targetLine;
}

double findProbability (vector<point> largeSet, double weights[])
{
    double numDisagreeing = 0;
    for (point p: largeSet)
    {
        if (p.score !=  calculateG(p, weights))
        {
            numDisagreeing++;
        }
    }
    return numDisagreeing / (largeSet.size());
}
bool isConverged (vector<point> points, double weights[] )
{
    for (point p : points)
    {
        if (p.score !=  calculateG(p, weights))
            return false;
    }
    return true;
}
int setScore (point currentPoint, line targetLine, point targetPoint)
{
    double slope = targetLine.slope;
    if ((currentPoint.y - targetPoint.y) > slope * (currentPoint.x - targetPoint.x))
        return 1;
    else
        return -1;
}
int main()
{

    double numTrials = 1000.0;
    srand(time(NULL));
    vector <point> samplePoints;
    vector <point> hellaPointsForProbability;
    double totalIterations = 0;
    double totalProbabilityOut = 0;
    double totalProbabilityIn = 0;
    vector<double> weights_0;
    vector<double> weights_1;
    vector<double> weights_2;
    for (int j = 0; j < (int)numTrials ; j++)
    {
        numIterations = 0;
        double weightVector[3] = {0.0 , 0.0, 0.0};
        samplePoints.clear();
        hellaPointsForProbability.clear();
        point startingPoint1;
        point startingPoint2;
        line targetLine;
        startingPoint1.x = returnRandomDouble(MIN, MAX);
        startingPoint1.y = returnRandomDouble(MIN, MAX);
        startingPoint2.x = returnRandomDouble(MIN, MAX);
        startingPoint2.y = returnRandomDouble(MIN, MAX);
        targetLine = returnTargetFunction(startingPoint1, startingPoint2);

        for (int k = 0; k < 1000; k++)
        {
        point currentPoint;
            currentPoint.x = returnRandomDouble(MIN, MAX);
            currentPoint.y = returnRandomDouble(MIN, MAX);
            currentPoint.score = setScore(currentPoint, targetLine, startingPoint1);
            hellaPointsForProbability.push_back(currentPoint);
        }
        int matrixRow = 0;
        mat X (NUMTRAININGPOINTS, 3);
        mat Y (NUMTRAININGPOINTS, 1);
        for (int i = 0 ; i < NUMTRAININGPOINTS ; i++)
        {
            point currentPoint;
            currentPoint.x = returnRandomDouble(MIN, MAX);
            currentPoint.y = returnRandomDouble(MIN, MAX);
            currentPoint.score = setScore(currentPoint, targetLine, startingPoint1);
            X(matrixRow, 0) = 1;
            X(matrixRow, 1) = currentPoint.x;
            X(matrixRow, 2) = currentPoint.y;
            Y(matrixRow, 0) = currentPoint.score;
            matrixRow++;
            samplePoints.push_back(currentPoint);
        }

        mat pseudoInverse = inv(X.t() * X) * X.t();
        mat regressionMatrix = pseudoInverse * Y;
        weightVector[0] = regressionMatrix(0,0);
        weightVector[1] = regressionMatrix(1,0);
        weightVector[2] = regressionMatrix(2,0);
        weights_0.push_back(weightVector[0]);
        weights_1.push_back(weightVector[1]);
        weights_2.push_back(weightVector[2]);
        totalProbabilityIn += findProbability(samplePoints, weightVector);
        totalProbabilityOut += findProbability(hellaPointsForProbability, weightVector);


        while (!isConverged(samplePoints, weightVector))
         {
            runPLA(samplePoints, weightVector);
        }
        totalIterations += numIterations;
    }
    cout << "Average in-sample probability (problem 5): "<<(totalProbabilityIn)/ (double)numTrials << endl;
    cout << "Average out-of-sample probability (problem 6): "<<(totalProbabilityOut)/ (double)numTrials << endl;

    cout << "Average number of iterations to converge usinig PLA (problem 7): " << totalIterations / (double) numTrials << endl;
    return 0;
}
