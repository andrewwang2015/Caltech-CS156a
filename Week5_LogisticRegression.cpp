#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <algorithm>

using namespace std;

#define MIN -1.0
#define MAX 1.0
#define NUMTRAININGPOINTS  100
#define NUMBEROFRUNS 100
#define LEARNINGRATE 0.01
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


double returnGradient(point p, double weights[])
{

	double innerProd = weights[0] + weights[1] * p.x + weights[2] * p.y;
	double gradient = log(1 + exp(-1 * p.score * innerProd));
	return gradient;

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
	for (point p : largeSet)
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

void updateWeightVector (double weights[], vector<point> points, double learningRate)
{
	for (point p : points)
	{
		double factor = returnGradient(p, weights);
		weights[0] -= learningRate * factor;
		weights[1] -= learningRate * factor;
		weights[2] -= learningRate * factor;
	}

}

int setScore (point currentPoint, line targetLine, point targetPoint)
{
	double slope = targetLine.slope;
	if ((currentPoint.y - targetPoint.y) > slope * (currentPoint.x - targetPoint.x))
		return 1;
	else
		return -1;
}

double returnCrossEntropy(double weights[], vector<point> points)
{
	double totalError = 0;
	for (point p : points)
	{
		double innerProd = weights[0] + weights[1] * p.x + weights[2] * p.y;
		totalError += log(1 + exp(-1 * p.score * innerProd));
	}
	return totalError / points.size();
}

double getDifference(double weights1[], double weights2[])
{
	double magnitude = 0;
	for (int i = 0 ; i < 3; i++)
		magnitude += pow(weights1[i] - weights2[i], 2.0);
	return sqrt(magnitude);
}

int main()
{
	srand(time(NULL));
	vector <point> samplePoints;
	vector <point> hellaPointsForProbability;
	double totalProbability = 0;
	double totalEpochs = 0;

	for (int x = 0; x < NUMBEROFRUNS; x++)
	{
		double epochs = 0;
		samplePoints.clear();
		hellaPointsForProbability.clear();
		double weights [] = {0, 0, 0};
		double newWeights[3];

		point startingPoint1;
		point startingPoint2;
		line targetLine;
		startingPoint1.x = returnRandomDouble(MIN, MAX);
		startingPoint1.y = returnRandomDouble(MIN, MAX);
		startingPoint2.x = returnRandomDouble(MIN, MAX);
		startingPoint2.y = returnRandomDouble(MIN, MAX);
		targetLine = returnTargetFunction(startingPoint1, startingPoint2);

		for (int k = 0; k < NUMTRAININGPOINTS; k++)
		{
			point currentPoint;
			currentPoint.x = returnRandomDouble(MIN, MAX);
			currentPoint.y = returnRandomDouble(MIN, MAX);
			currentPoint.score = setScore(currentPoint, targetLine, startingPoint1);
			samplePoints.push_back(currentPoint);
		}
		double difference = 0;
		do
		{
			random_shuffle(samplePoints.begin(), samplePoints.end());
			newWeights[0] = weights[0];
			newWeights[1] = weights[1];
			newWeights[2] = weights[2];
			updateWeightVector(weights, samplePoints, LEARNINGRATE);
			difference = getDifference(newWeights, weights);
			epochs++;
		} while (difference >= 0.01);

		for (int j = 0; j < 10000; j++)
		{
			point currentPoint;
			currentPoint.x = returnRandomDouble(MIN, MAX);
			currentPoint.y = returnRandomDouble(MIN, MAX);
			currentPoint.score = setScore(currentPoint, targetLine, startingPoint1);
			hellaPointsForProbability.push_back(currentPoint);
		}
		totalEpochs += epochs;
		totalProbability += findProbability(hellaPointsForProbability, weights);
	}
	cout << totalEpochs / NUMBEROFRUNS << endl;
	cout << totalProbability / NUMBEROFRUNS << endl;

	return 0;
}
