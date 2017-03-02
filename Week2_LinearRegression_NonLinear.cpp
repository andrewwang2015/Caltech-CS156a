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
#define NUMTRAININGPOINTS  1000

int numIterations = 0;

typedef struct _point
{
    double x;
    double y;
    int score;
} point;

int setScore (point currentPoint)
{
    double value = currentPoint.x * currentPoint.x + currentPoint.y * currentPoint.y - 0.6;
    if (value > 0)
        return 1;
    return -1;
}

int calculateG (point p, double weights[])
{
    double innerProd = weights[0] + weights[1] * p.x + weights[2] * p.y + weights[3] * p.x * p.y + weights[4] * p.x * p.x + weights[5] * p.y * p.y;
    if (innerProd > 0)
        return 1;
    else
        return -1;
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

double returnRandomDouble(double min, double max)
{
    double randomNum = (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
    return randomNum;
}

double returnAverage(vector<double> set)
{
    double sum = 0;
    for (double i : set)
    {
        sum += i;
    }
    return (sum / set.size());
}

int flipScore(int value)
{
    return -1 * value;
}

int returnRandomInteger (int min, int max)
{
    int randNum = rand()%(max-min + 1) + min;
    return randNum;
}
int main()
{

    double numTrials = 1000.0;
    srand(time(NULL));
    vector <point> samplePoints;
    vector <point> hellaPointsForProbability;
    double totalIterations = 0;
    double totalProbabilityOut = 0;
    vector<double> weights_0;
    vector<double> weights_1;
    vector<double> weights_2;
    vector<double> weights_3;
    vector<double> weights_4;
    vector<double> weights_5;
    for (int j = 0; j < (int)numTrials ; j++)
    {
        numIterations = 0;
        double weightVector[6] = {0.0 , 0.0, 0.0, 0.0, 0.0, 0.0};
        samplePoints.clear();
        hellaPointsForProbability.clear();

        for (int k = 0; k < 1000; k++)
        {
            point currentPoint;
            int flipOrNot = returnRandomInteger(0, 9);
            currentPoint.x = returnRandomDouble(MIN, MAX);
            currentPoint.y = returnRandomDouble(MIN, MAX);
            int score = setScore(currentPoint);
            if (flipOrNot == 5)
                score = flipScore(score);
            currentPoint.score = score;
            hellaPointsForProbability.push_back(currentPoint);
        }
        int matrixRow = 0;
        mat X (NUMTRAININGPOINTS, 6);
        mat Y (NUMTRAININGPOINTS, 1);
        for (int i = 0 ; i < NUMTRAININGPOINTS ; i++)
        {
            int flipOrNot = returnRandomInteger(0, 9);
            point currentPoint;
            currentPoint.x = returnRandomDouble(MIN, MAX);
            currentPoint.y = returnRandomDouble(MIN, MAX);
            int score = setScore(currentPoint);
            if (flipOrNot == 5)
                score = flipScore(score);
            currentPoint.score = score;
            X(matrixRow, 0) = 1;
            X(matrixRow, 1) = currentPoint.x;
            X(matrixRow, 2) = currentPoint.y;
            X(matrixRow, 3) = currentPoint.x * currentPoint.y;
            X(matrixRow, 4) = currentPoint.x * currentPoint.x;
            X(matrixRow, 5) = currentPoint.y * currentPoint.y;
            Y(matrixRow, 0) = currentPoint.score;
            matrixRow++;
            samplePoints.push_back(currentPoint);
        }

        mat pseudoInverse = inv(X.t() * X) * X.t();

        mat regressionMatrix = pseudoInverse * Y;
        //cout << regressionMatrix << endl;
        weightVector[0] = regressionMatrix(0, 0);
        weightVector[1] = regressionMatrix(1, 0);
        weightVector[2] = regressionMatrix(2, 0);
        weightVector[3] = regressionMatrix(3, 0);
        weightVector[4] = regressionMatrix(4, 0);
        weightVector[5] = regressionMatrix(5, 0);
        weights_0.push_back(weightVector[0]);
        weights_1.push_back(weightVector[1]);
        weights_2.push_back(weightVector[2]);
        weights_3.push_back(weightVector[3]);
        weights_4.push_back(weightVector[4]);
        weights_5.push_back(weightVector[5]);
        totalProbabilityOut += findProbability(hellaPointsForProbability, weightVector);
    }
    cout << returnAverage(weights_0) << endl;
    cout << returnAverage(weights_1) << endl;
    cout << returnAverage(weights_2) << endl;
    cout << returnAverage(weights_3) << endl;
    cout << returnAverage(weights_4) << endl;
    cout << returnAverage(weights_5) << endl;
    cout << totalProbabilityOut/numTrials << endl;
    return 0;
}
