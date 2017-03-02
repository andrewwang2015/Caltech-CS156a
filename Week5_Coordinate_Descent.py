import math as m

def getError(weights):
    u = weights[0]
    v = weights[1]
    
    return (u * m.exp(v) - 2 * v * m.exp(-1 * u)) ** 2

def updateWeightsU(weights, learningRate):
    weights[0] = weights[0] - learningRate * returnGradientU(weights)

def updateWeightsV(weights, learningRate):
    weights[1] = weights[1] - learningRate * returnGradientV(weights)
    
def returnGradientU(weights):
    u = weights[0]
    v = weights[1]
    gradientU = 2 * (m.exp(v) + 2 * v * m.exp(-u)) * (u * m.exp(v) - 2 * v * m.exp(-u))
    return gradientU

def returnGradientV(weights):
    u = weights[0]
    v = weights[1]
    gradientV = 2 * (u * m.exp(v) - 2 * m.exp(-u)) * (u * m.exp(v)  - 2 * v * m.exp(-u))
    return gradientV   
    

def main():
    weights = [1.0, 1.0]
    learningRate = 0.1
    count = 0
    for i in range(15):
        updateWeightsU(weights, learningRate)
        updateWeightsV(weights, learningRate)
    print (getError(weights))

main()
        
        