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
#define k 3.0

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


int main()
{

    ifstream infile("in.dta.txt");
    double x;
    double y;
    double score;
    int inFileSize = 0;

    vector<point> inSamplePoints;
    while (infile >> x >> y >> score)
    {
        point currentPoint;
        currentPoint.x = x;
        currentPoint.y = y;
        currentPoint.score = (int)score;
        inSamplePoints.push_back(currentPoint);
        inFileSize++;
    }

    ifstream outfile("out.dta.txt");

    int outFileSize = 0;
    vector<point> outSamplePoints;
    while (outfile >> x >> y >> score)
    {
        point currentPoint;
        currentPoint.x = x;
        currentPoint.y = y;
        currentPoint.score = (int)score;
        outSamplePoints.push_back(currentPoint);
        outFileSize++;
    }


    int matrixRow = 0;
    mat X (inSamplePoints.size(), 8);
    mat Y (inSamplePoints.size(), 1);
    for (point p : inSamplePoints)
    {
        X(matrixRow, 0) = 1;
        X(matrixRow, 1) = p.x;
        X(matrixRow, 2) = p.y;
        X(matrixRow, 3) = p.x * p.x;
        X(matrixRow, 4) = p.y * p.y;
        X(matrixRow, 5) = p.x * p.y;
        X(matrixRow, 6) = abs(p.x - p.y);
        X(matrixRow, 7) = abs(p.x + p.y);
        Y(matrixRow, 0) = p.score;
        matrixRow++;
    }

    mat pseudoInverse = inv(X.t() * X) * X.t();
    mat regressionMatrix = pseudoInverse * Y;
    double weightVector[8];
    weightVector[0] = regressionMatrix(0, 0);
    weightVector[1] = regressionMatrix(1, 0);
    weightVector[2] = regressionMatrix(2, 0);
    weightVector[3] = regressionMatrix(3, 0);
    weightVector[4] = regressionMatrix(4, 0);
    weightVector[5] = regressionMatrix(5, 0);
    weightVector[6] = regressionMatrix(6, 0);
    weightVector[7] = regressionMatrix(7, 0);

    double totalProbabilityIn = findProbability(inSamplePoints, weightVector);
    double totalProbabilityOut = findProbability(outSamplePoints, weightVector);
    cout << "Average in-sample probability using unconstrained linear regression (problem 2): " << (totalProbabilityIn) << endl;
    cout << "Average out-of-sample probability using unconstrained linear regression (problem 2): " << (totalProbabilityOut) << endl;

    mat identity(8, 8);
    double lambda = pow(10, k);
    mat pseudoInverse_mod = inv(X.t() * X + lambda * identity.eye());
    mat regRegressionMatrix = pseudoInverse_mod * (X.t() * Y);
    double weightVectorReg[8];
    weightVectorReg[0] = regRegressionMatrix(0, 0);
    weightVectorReg[1] = regRegressionMatrix(1, 0);
    weightVectorReg[2] = regRegressionMatrix(2, 0);
    weightVectorReg[3] = regRegressionMatrix(3, 0);
    weightVectorReg[4] = regRegressionMatrix(4, 0);
    weightVectorReg[5] = regRegressionMatrix(5, 0);
    weightVectorReg[6] = regRegressionMatrix(6, 0);
    weightVectorReg[7] = regRegressionMatrix(7, 0);

    double totalProbabilityInReg = findProbability(inSamplePoints, weightVectorReg);
    double totalProbabilityOutReg = findProbability(outSamplePoints, weightVectorReg);
    cout << "Average in-sample probability using weight decay (problem 3): " << (totalProbabilityInReg) << endl;
    cout << "Average out-of-sample probability using weight decay (problem 3): " << (totalProbabilityOutReg) << endl;

    double minOutError = 999;
    double minK = 0;
    vector<double> possibleKs = {2, 1, 0, -1, -2};
    for (double i : possibleKs)
    {
        mat identity(8, 8);
        double lambda = pow(10, i);
        mat pseudoInverse_mod = inv(X.t() * X + lambda * identity.eye());
        mat regRegressionMatrix = pseudoInverse_mod * (X.t() * Y);
        double weightVectorReg[8];
        weightVectorReg[0] = regRegressionMatrix(0, 0);
        weightVectorReg[1] = regRegressionMatrix(1, 0);
        weightVectorReg[2] = regRegressionMatrix(2, 0);
        weightVectorReg[3] = regRegressionMatrix(3, 0);
        weightVectorReg[4] = regRegressionMatrix(4, 0);
        weightVectorReg[5] = regRegressionMatrix(5, 0);
        weightVectorReg[6] = regRegressionMatrix(6, 0);
        weightVectorReg[7] = regRegressionMatrix(7, 0);
        double totalProbabilityOutReg = findProbability(outSamplePoints, weightVectorReg);
        if (totalProbabilityOutReg < minOutError)
        {
            minOutError = totalProbabilityOutReg;
            minK = i;
        }
    }
    cout << "Smallest out-of-sample probability is: " << minOutError << " corresponding to k = " << minK << endl;

    minOutError = 999;
    minK = 0;
    for (int i = -20; i <= 20; i++)
    {
        mat identity(8, 8);
        double lambda = pow(10, i);
        mat pseudoInverse_mod = inv(X.t() * X + lambda * identity.eye());
        mat regRegressionMatrix = pseudoInverse_mod * (X.t() * Y);
        double weightVectorReg[8];
        weightVectorReg[0] = regRegressionMatrix(0, 0);
        weightVectorReg[1] = regRegressionMatrix(1, 0);
        weightVectorReg[2] = regRegressionMatrix(2, 0);
        weightVectorReg[3] = regRegressionMatrix(3, 0);
        weightVectorReg[4] = regRegressionMatrix(4, 0);
        weightVectorReg[5] = regRegressionMatrix(5, 0);
        weightVectorReg[6] = regRegressionMatrix(6, 0);
        weightVectorReg[7] = regRegressionMatrix(7, 0);
        double totalProbabilityOutReg = findProbability(outSamplePoints, weightVectorReg);
        cout << totalProbabilityOutReg << ", " 
        if (totalProbabilityOutReg < minOutError)
        {
            minOutError = totalProbabilityOutReg;
            minK = i;
        }
    }
    cout << "Smallest out-of-sample probability from [-20, 20] is: " << minOutError << " corresponding to k = " << minK << endl;

    return 0;
}

