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


double returnRandomDouble(double min, double max)
{
	double randomNum = (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
	return randomNum;
}


vector<double> returnGradient(point p, double weights[])
{
	vector<double> gradient;
	double innerProd = weights[0] + weights[1] * p.x + weights[2] * p.y;
	double dividor = 1 + exp(p.score * innerProd);
	gradient.push_back(-1 * p.score / dividor);
	gradient.push_back(-1 * p.score * p.x / dividor);
	gradient.push_back(-1 * p.score * p.y / dividor);
	return gradient;

}

line returnTargetFunction(point point1, point point2)
{
	line targetLine;
	double targetSlope = ((point2.y) - (point1.y)) / ((point2.x) - (point1.x));
	targetLine.slope = targetSlope;
	return targetLine;
}



void updateWeightVector (double weights[], point p, double learningRate)
{

	vector<double> factor = returnGradient(p, weights);
	weights[0] -= learningRate * factor[0];
	weights[1] -= learningRate * factor[1];
	weights[2] -= learningRate * factor[2];


}

int setScore (point currentPoint, line targetLine, point targetPoint)
{
	double slope = targetLine.slope;
	if ((currentPoint.y - targetPoint.y) > slope * (currentPoint.x - targetPoint.x))
		return 1;
	else
		return -1;
}

double returnCrossEntropy(double weights[], point p)
{


	double innerProd = weights[0] + weights[1] * p.x + weights[2] * p.y;
	double error = log(1 + exp(-1 * p.score * innerProd));
	return error;

}

double getDifference(double weights1[], double weights2[])
{

	return sqrt((weights1[0] - weights2[0]) * (weights1[0] - weights2[0]) + (weights1[1] - weights2[1]) * (weights1[1] - weights2[1]) +  (weights1[2] - weights2[2]) * (weights1[2] - weights2[2]) );
}

int main()
{
	srand(time(NULL));
	vector <point> samplePoints;
	vector <point> hellaPointsForProbability;
	double totalError = 0;
	double totalEpochs = 0;

	for (int x = 0; x < NUMBEROFRUNS; x++)
	{
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

		for (int k = 0; k < NUMTRAININGPOINTS + 1000; k++)
		{
			point currentPoint;
			currentPoint.x = returnRandomDouble(MIN, MAX);
			currentPoint.y = returnRandomDouble(MIN, MAX);
			currentPoint.score = setScore(currentPoint, targetLine, startingPoint1);
			samplePoints.push_back(currentPoint);
		}
		double difference = 0;
		newWeights[0] = weights[0];
		newWeights[1] = weights[1];
		newWeights[2] = weights[2];
		int count = 0;
		do
		{
			weights[0] = newWeights[0];
			weights[1] = newWeights[1];
			weights[2] = newWeights[2];
			random_shuffle(samplePoints.begin(), samplePoints.end());
			for (int y = 0; y < NUMTRAININGPOINTS; y++)
			{
				updateWeightVector(newWeights, samplePoints[y], LEARNINGRATE);
			}
			difference = getDifference(newWeights, weights);
			totalEpochs++;
		} while (difference >= 0.01);

		random_shuffle(samplePoints.begin(), samplePoints.end());
		double error = 0;
		for (int j = 0; j < 1000; j++)
		{
			error += (returnCrossEntropy(newWeights, samplePoints[j]));
		}
		totalError += error/1000;
		

	}
	cout << totalEpochs / NUMBEROFRUNS << endl;
	cout << totalError / NUMBEROFRUNS << endl;

	return 0;
}
