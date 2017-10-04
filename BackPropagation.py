
import math 

import random

def train_test_split(input_array, prediction_array, test_size):
    if len(input_array) != len(prediction_array):
        raise Exception('length of input and prediction does not match')

    length_array = len(input_array)
    list_random = random.sample(range(0, length_array), int(length_array * test_size / 100))
    train_input = []
    test_input = []
    train_prediction = []
    test_prediction = []

    list_index = 0;
    for index in range(0, length_array):
        if (list_index < len(list_random)) and (list_random[list_index] == index):
            test_input.append(input_array[index])
            test_prediction.append(prediction_array[index])
            list_index += 1
        else:
            train_input.append(input_array[index])
            train_prediction.append(prediction_array[index])

    return train_input, test_input, train_prediction, test_prediction

class Activate:
    
    def activationFunction(self,input):
        return 1/(1+math.exp(-input))

a = Activate()

print(a.activationFunction(2))

X = [[0, 0], [0, 1], [1, 0], [1, 1], [1,1]] * 2
y = [0, 1, 1, 0, 1] * 2
X_train, X_test, y_train, y_test = train_test_split(X, y, 20)

print(X_train)
print(X_test)

print(y_train)
print(y_test)

