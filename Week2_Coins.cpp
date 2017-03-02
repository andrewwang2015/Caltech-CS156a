#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <time.h>
using namespace std;


double v1 = 0;
double v_rand = 0;
double v_min = 0;

class Coin
{
public: 
	int totalHeads; // heads = 1, tails = 0
	double fraction;
	Coin()
	{
		totalHeads = 0;
		fraction = 0.0;
	}
};

void runExperiment()
{
	srand(time(NULL));
	vector <Coin> coinCollection;
	coinCollection.clear(); 
	int min = 10;
	int minIndex = 0;
	for (int i = 0 ; i < 1000 ; i++)
	{
		Coin currentCoin;
		for (int j = 0 ; j < 10; j++)
		{
			int randomNum = rand() % 2;
			if (randomNum == 1)
			{
				currentCoin.totalHeads += 1;
			}
		}
		if (currentCoin.totalHeads < min)
		{
			min = currentCoin.totalHeads;
			minIndex = i;
		}
		currentCoin.fraction = currentCoin.totalHeads/ 10.0;
		coinCollection.push_back(currentCoin);
	}
	Coin firstCoin = coinCollection[0];
	int random = rand() % 1000;
	Coin randomCoin = coinCollection[random];
	Coin leastCoin = coinCollection[minIndex];
	v1 += firstCoin.fraction;
	v_rand += randomCoin.fraction;
	v_min += leastCoin.fraction;
}

int main()
{
	for (int k = 0; k < 100000; k++)
	{
		runExperiment();
	}
	cout << "First coin: " << v1/100000.0 << endl;
	cout << "Random coin: " << v_rand/100000.0 << endl;
	cout << "Minimum coin: " << v_min/100000.0 << endl;
	return 0;
}