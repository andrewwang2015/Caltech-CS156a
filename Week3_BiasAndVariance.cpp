#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <armadillo>
#include <algorithm>
#include <math.h>
#include <cmath>
using namespace std;
using namespace arma;
#define MIN -1.0
#define MAX 1.0
#define NUMTRAININGPOINTS  100000
#define _USE_MATH_DEFINES

int numIterations = 0;

typedef struct _point
{
    double x;
    double y;
    int score;
} point;

double returnBestSlopeSquared(point point1, point point2)
{
    double minError = 9999;
    double currentSlope = -3.0;
    double maxSlopeTesting = 3.0;
    double bestSlope = 0;

    while (currentSlope < maxSlopeTesting)
    {
        double y1 = point1.x * point1.x * currentSlope;
        double y2 = point2.x * point2.x * currentSlope;

        double error1 = pow(point1.y - y1, 2);
        double error2 = pow(point2.y - y2, 2);
        double averageError = (error1 + error2) / 2;
        if (averageError < minError)
        {
            minError = averageError;
            bestSlope = currentSlope;
        }
        currentSlope += 0.1;
    }
    return bestSlope;

}

double returnBestSlope(point point1, point point2)
{
    double minError = 9999;
    double currentSlope = -3.0;
    double maxSlopeTesting = 3.0;
    double bestSlope = 0;

    while (currentSlope < maxSlopeTesting)
    {
        double y1 = point1.x * currentSlope;
        double y2 = point2.x * currentSlope;

        double error1 = pow(point1.y - y1, 2);
        double error2 = pow(point2.y - y2, 2);
        double averageError = (error1 + error2) / 2;
        if (averageError < minError)
        {
            minError = averageError;
            bestSlope = currentSlope;
        }
        currentSlope += 0.1;
    }
    return bestSlope;

}

double returnAverageBias(double bestSlope)
{
    double totalBias = 0.0;
    int count = 0;
    for (double i = MIN; i <= MAX; i += 0.01)
    {
        double currentBias = pow(bestSlope * i - sin(M_PI * i), 2);
        totalBias += currentBias;
        count++;
    }
    return totalBias / count;
}

double returnAverageBiasSquared(double bestSlope)
{
    double totalBias = 0.0;
    int count = 0;
    for (double i = MIN; i <= MAX; i += 0.01)
    {
        double currentBias = pow(bestSlope * i * i - sin(M_PI * i), 2);
        totalBias += currentBias;
        count++;
    }
    return totalBias / count;
}

double returnAverageBiasSquaredConstant(double bestb, double besta)
{
    double totalBias = 0.0;
    int count = 0;
    for (double i = MIN; i <= MAX; i += 0.01)
    {
        double currentBias = pow(besta * i * i + bestb - sin(M_PI * i), 2);
        totalBias += currentBias;
        count++;
    }
    return totalBias / count;
}

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


double returnRandomDouble(double min, double max)
{
    double randomNum = (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
    return randomNum;
}

double returnAverageVariance(vector<double> g, double bestSlope)
{
    double totalVariance = 0;
    int count = 0;
    for (double i = MIN; i <= MAX; i += 0.01)
    {
        totalVariance += pow(g[count] * i - bestSlope * i, 2);
        count ++;
    }

    return totalVariance / count;
}

double returnAverageVarianceSquared(vector<double> g, double bestSlope)
{
    double totalVariance = 0;
    int count = 0;
    for (double i = MIN; i <= MAX; i += 0.01)
    {
        totalVariance += pow(g[count] * i * i - bestSlope * i * i, 2);
        count ++;
    }

    return totalVariance / count;
}

double returnAverageVarianceSquaredWithConstant(vector<double> b, vector<double> a, double besta, double bestb)
{
    double totalVariance = 0;
    int count = 0;
    for (double i = MIN; i <= MAX; i += 0.01)
    {
        totalVariance += pow(b[count] + a[count]* i * i - (bestb + besta * i * i), 2);
        count ++;
    }

    return totalVariance / count;
}

int returnRandomInteger (int min, int max)
{
    int randNum = rand() % (max - min + 1) + min;
    return randNum;
}
int main()
{
    vector<double> g_Ds;
    vector<double> g_Ds_squared;
    srand(time(NULL));
    double totalSlope = 0;
    for (int j = 0; j < NUMTRAININGPOINTS; j++)
    {

        point currentPoint1;
        point currentPoint2;
        currentPoint1.x = returnRandomDouble(MIN, MAX);
        currentPoint1.y = sin(M_PI * currentPoint1.x);
        currentPoint2.x = returnRandomDouble(MIN, MAX);
        currentPoint2.y = sin(M_PI * currentPoint2.x);
        double currentBestSlope = returnBestSlope(currentPoint1, currentPoint2);
        g_Ds.push_back(currentBestSlope);
        totalSlope += currentBestSlope;

    }

    double totalSlopeSquared = 0;

    for (int j = 0; j < NUMTRAININGPOINTS; j++)
    {

        point currentPoint1;
        point currentPoint2;
        currentPoint1.x = returnRandomDouble(MIN, MAX);
        currentPoint1.y = sin(M_PI * currentPoint1.x);
        currentPoint2.x = returnRandomDouble(MIN, MAX);
        currentPoint2.y = sin(M_PI * currentPoint2.x);
        double currentBestSlopeSquared = returnBestSlopeSquared(currentPoint1, currentPoint2);
        g_Ds_squared.push_back(currentBestSlopeSquared);
        totalSlopeSquared += currentBestSlopeSquared;

    }
    mat X (2, 2);
    mat Y (2, 1);
    double totalWeights0 = 0;
    double totalWeights1= 0;
    vector<double> weights_0;
    vector<double> weights_1;
    double weightVector[2] = {0.0, 0.0};
    for (int j = 0; j < NUMTRAININGPOINTS; j++)
    {

        point currentPoint1;
        point currentPoint2;
        currentPoint1.x = returnRandomDouble(MIN, MAX);
        currentPoint1.y = sin(M_PI * currentPoint1.x);
        currentPoint2.x = returnRandomDouble(MIN, MAX);
        currentPoint2.y = sin(M_PI * currentPoint2.x);

        X(0, 0) = 1;
        X(0, 1) = currentPoint1.x * currentPoint1.x;
        X(1, 0) = 1;
        X(1, 1) = currentPoint2.x * currentPoint2.x;
        Y(0, 0) = currentPoint1.y;
        Y(1, 0) = currentPoint2.y;
        mat pseudoInverse = inv(X.t() * X) * X.t();
        mat regressionMatrix = pseudoInverse * Y;
        weightVector[0] = regressionMatrix(0, 0);
        weightVector[1] = regressionMatrix(1, 0);
        weights_0.push_back(weightVector[0]);
        weights_1.push_back(weightVector[1]);
        totalWeights0 += weightVector[0];
        totalWeights1 += weightVector[1];
    }

    cout << "Average best slope (ax): " << (totalSlope / NUMTRAININGPOINTS) << endl;
    cout << "Average bias (ax): " << returnAverageBias(totalSlope / NUMTRAININGPOINTS) << endl;
    cout << "Average variance (ax): " << returnAverageVariance(g_Ds, totalSlope / NUMTRAININGPOINTS) << endl;

    cout << "Average best slope (ax^2): " << (totalSlopeSquared / NUMTRAININGPOINTS) << endl;
    cout << "Average bias (ax^2): " << returnAverageBiasSquared(totalSlopeSquared / NUMTRAININGPOINTS) << endl;
    cout << "Average variance (ax^2): " << returnAverageVarianceSquared(g_Ds_squared, totalSlopeSquared / NUMTRAININGPOINTS) << endl;

    cout << "Average a, b (ax^2 + b): " << (totalWeights1 / NUMTRAININGPOINTS) << ", " << (totalWeights0/ NUMTRAININGPOINTS) << endl;
    cout << "Average bias (ax^2 + b): " << returnAverageBiasSquaredConstant((totalWeights0/ NUMTRAININGPOINTS) ,totalWeights1 / NUMTRAININGPOINTS) << endl;
    cout << "Average variance (ax^2 + b): " << returnAverageVarianceSquaredWithConstant(weights_0, weights_1, (totalWeights0/ NUMTRAININGPOINTS) ,totalWeights1 / NUMTRAININGPOINTS) << endl;
    return 0;
}
