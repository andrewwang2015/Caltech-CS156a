#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <armadillo>
#include <algorithm>
#include <fstream>
#include <cmath>

using namespace std;
using namespace arma;
#define MIN -1.0
#define MAX 1.0


int numIterations = 0;

typedef struct _point
{
    double x;
    double y;
    int score;
} point;

int calculateG (point p, double weights[])
{
    double innerProd = weights[0] + weights[1] * p.x + weights[2] * p.y + weights[3] * p.x * p.x + weights[4] * p.y * p.y + weights[5] * p.x * p.y + weights[6] * abs(p.x - p.y) + weights[7] * abs(p.x + p.y);
    if (innerProd > 0)
        return 1;
    else
        return -1;
}

double findProbability (vector<point> largeSet, double weights[])
{
    double numDisagreeing = 0;
    for (point p : largeSet)
    {
        if (p.score !=  calculateG(p, weights))
        {
            numDisagreeing++;
        }
    }
    return numDisagreeing / (largeSet.size());
}

double evaluateSquaredError(vector<point> points, double slope, double intercept)
{
    double avgError = 0;
    for (point p: points)
    {
        double estimate = p.x * slope + intercept;
        avgError += pow(p.score - estimate, 2);

    }
    return avgError/3;
}



double evaluateSquaredErrorC(vector<point> points, double intercept)
{
    double avgError = 0;
    for (point p: points)
    {
        double estimate = intercept;
        avgError += pow(p.score - estimate, 2);

    }
    return avgError/3;
}

int main()
{
    vector<point> points;
    double x;
    double y;
    double score;
    point p;
    p.x = -1;
    p.score = 0;

    points.push_back(p);

    point p1;
    p1.x = 1;
    p1.score = 0;

    points.push_back(p1);

    point p2;
    p2.x = 2.39417;
    p2.score = 1;
    points.push_back(p2);

    int matrixRow = 0;
    mat X (3, 1);
    mat Y (3, 1);
    for (point p : points)
    {
        Y(matrixRow, 0) = p.score;
        X(matrixRow, 0) = 1;
        matrixRow++;
    }
    mat pseudoInverse = inv(X.t() * X) * X.t();
    mat regressionMatrix = pseudoInverse * Y;
    double weights[1];
    for (int i = 0; i < 1; i++) {
        weights[i] = regressionMatrix(i, 0);
    }

    matrixRow = 0;

    mat X1 (3, 2);
    mat Y1 (3, 1);
    for (point p : points)
    {
        Y1(matrixRow, 0) = p.score;
        X1(matrixRow, 0) = 1;
        X1(matrixRow, 1) = p.x;
        matrixRow++;

    }
    pseudoInverse = inv(X1.t() * X1) * X1.t();
    regressionMatrix = pseudoInverse * Y1;
    double weights1[2];
    for (int i = 0; i < 2; i++) {
        weights1[i] = regressionMatrix(i, 0);
    }

    cout << "Error from constant model: " << evaluateSquaredErrorC(points, weights[0]) << endl;
    cout << "Error from linear model: " << evaluateSquaredError(points, weights[1], weights[0]) << endl;
    return 0;
}

