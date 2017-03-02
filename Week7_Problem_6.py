import random
e1avg = 0
e2avg = 0
emin = 0

for i in range(10000):
    e1 = random.random()
    e2 = random.random()
    e1avg += e1
    e2avg += e2
    emin += min(e1, e2)

print ("Expected value of e1: " + str(e1avg/10000))
print ("Expected value of e2: " + str(e2avg/10000))
print ("Min: " + str(emin/10000))
