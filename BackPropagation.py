
import math 

class Activate:
    
    def activationFunction(self,input):
        return 1/(1+math.exp(-input))
a = Activate()

print(a.activationFunction(2))
