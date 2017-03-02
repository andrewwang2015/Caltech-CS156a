#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <time.h>
using namespace std;

#define MIN -1.0
#define MAX 1.0
#define NUMTRAININGPOINTS  100

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
	srand(time(NULL));
	vector <point> samplePoints;
	vector <point> hellaPointsForProbability;
	double totalIterations = 0;
	double totalProbability = 0;
	for (int j = 0; j < 1000 ; j++)
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

		for (int k = 0; k < 100000; k++)
		{
			point currentPoint;
			currentPoint.x = returnRandomDouble(MIN, MAX);
			currentPoint.y = returnRandomDouble(MIN, MAX);
			currentPoint.score = setScore(currentPoint, targetLine, startingPoint1);
			hellaPointsForProbability.push_back(currentPoint);
		}
		for (int i = 0 ; i < NUMTRAININGPOINTS ; i++)
		{
			point currentPoint;
			currentPoint.x = returnRandomDouble(MIN, MAX);
			currentPoint.y = returnRandomDouble(MIN, MAX);
			currentPoint.score = setScore(currentPoint, targetLine, startingPoint1);
			samplePoints.push_back(currentPoint);
		}


		while (!isConverged(samplePoints, weightVector))
		{
			runPLA(samplePoints, weightVector);
		}
		double currentProbability = findProbability(hellaPointsForProbability, weightVector);
		totalProbability += currentProbability;
		totalIterations += numIterations;
	}

	cout << totalIterations / 1000 << endl;
	cout << totalProbability / 1000 << endl;
	return 0;
}
