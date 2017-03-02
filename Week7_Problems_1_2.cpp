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

    vector<point> first25;
    for (int i = 0; i < 25; i++)
    {
        first25.push_back(inSamplePoints[i]);
    }
    vector<point> last10;
    for (int i = 25; i < inSamplePoints.size(); i++)
    {
        last10.push_back(inSamplePoints[i]);
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

    double minOutError = 999;
    int minK = 0;
    for (int k = 3; k <= 7; k++)
    {
        int matrixRow = 0;
        mat X (first25.size(), k + 1);
        mat Y (first25.size(), 1);
        for (point p : first25)
        {
            Y(matrixRow, 0) = p.score;
            X(matrixRow, 0) = 1;
            X(matrixRow, 1) = p.x;
            X(matrixRow, 2) = p.y;
            X(matrixRow, 3) = p.x * p.x;
            if (k == 3) {
                matrixRow++;
                continue;
            }
            X(matrixRow, 4) = p.y * p.y;
            if (k == 4) {
                matrixRow++;
                continue;
            }
            X(matrixRow, 5) = p.x * p.y;
            if (k == 5) {
                matrixRow++;
                continue;
            }
            X(matrixRow, 6) = abs(p.x - p.y);
            if (k == 6) {
                matrixRow++;
                continue;
            }
            X(matrixRow, 7) = abs(p.x + p.y);
            matrixRow++;

        }
        mat pseudoInverse = inv(X.t() * X) * X.t();
        mat regressionMatrix = pseudoInverse * Y;
        double weightVector[8];
        for (int i = 0; i < 8; i++) {
            if (i <= k)
                weightVector[i] = regressionMatrix(i, 0);
            else
                weightVector[i] = 0;
        }
        double totalProbabilityOutReg = findProbability(outSamplePoints , weightVector);
        if (totalProbabilityOutReg < minOutError)
        {
            minOutError = totalProbabilityOutReg;
            minK = k;
        }
    }
    cout << "Minimum out error: " << minOutError << ". Minimum k: " << minK << endl;
    return 0;
}

