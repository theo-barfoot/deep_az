bob = 1

def mytest(basd):
    return basd + 1

for i in range(1, 1000):
    bob = mytest(bob)

print(bob)