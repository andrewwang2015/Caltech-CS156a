import math as m

def getError(weights):
    u = weights[0]
    v = weights[1]
    
    return (u * m.exp(v) - 2 * v * m.exp(-1 * u)) ** 2

def updateWeights(weights, learningRate):
    change = returnGradient(weights)
    weights[0] = weights[0] - learningRate * change[0]
    weights[1] = weights[1] - learningRate * change[1]
    
def returnGradient(weights):
    gradientChange = []
    u = weights[0]
    v = weights[1]
    gradientU = 2 * (m.exp(v) + 2 * v * m.exp(-u)) * (u * m.exp(v) - 2 * v * m.exp(-u))
    gradientV = 2 * (u * m.exp(v) - 2 * m.exp(-u)) * (u * m.exp(v)  - 2 * v * m.exp(-u))
    gradientChange.append(gradientU)
    gradientChange.append(gradientV)
    return gradientChange

def main():
    weights = [1.0, 1.0]
    learningRate = 0.1
    count = 0
    while (getError(weights) >= 10** (-14)):
        updateWeights(weights, learningRate)
        count += 1
    print ("The number of iterations it takes to get an error less than 10^(-14) is " + str(count))
    print (weights)

main()
        
        