import random
from enum import Enum
from macpath import norm_error
import numpy as np
import math

class Layer_Type(Enum):
    INPUT_LAYER = 1
    HIDDEN_LAYER = 2
    OUTPUT_LAYER = 3

class Node:
    def __init__(self, number_of_weights, value = 0):
        self.weights = []
        self.value = value
        for index in range(0, number_of_weights):
            self.weights.append(random.random())

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_weight(self, index, weight):
        self.weights[index] = weight

    def get_weight(self, index):
        return self.weights[index]

class Layer:
    def __init__(self, nof_nodes, nof_nodes_next_layer, layer_type):
        self.nodes = []
        self.nof_nodes = nof_nodes
        self.nof_nodes_next_layer = nof_nodes_next_layer
        self.layer_type = layer_type

        #include extra node for bias in each layer
        self.nodes.append(Node(nof_nodes_next_layer, 1))

        for index in range(0, nof_nodes):
            if layer_type != Layer_Type.OUTPUT_LAYER:
                self.nodes.append(Node(nof_nodes_next_layer))
            else:
                self.nodes.append(Node(0))

    def set_weight(self, i, j, weight):
        self.nodes[i].set_weight(j, weight)

    def set_node_value(self, index, value):
        self.nodes[index].set_value(value)

    def print(self):
        print("Number of nodes: %d" % (self.nof_nodes))
        for node in self.nodes:
            print("node value: %f" % (node.value))
            for weight in node.weights:
                print(weight)

class NN:
    def __init__(self, X_train, y_train, max_iterations, *nof_nodes):
        self.nof_hiden_layers = len(nof_nodes)
        self.layers = []

        #input layer
        self.layers.append(Layer(np.shape(X_train)[1], nof_nodes[0], Layer_Type.INPUT_LAYER))

        #hidden layers
        for index in range(0, self.nof_hiden_layers - 1):
            self.layers.append(Layer(nof_nodes[index], nof_nodes[index+1], Layer_Type.HIDDEN_LAYER))

        self.layers.append(Layer(nof_nodes[self.nof_hiden_layers - 1], 1, Layer_Type.HIDDEN_LAYER))

        #output layer
        #self.layers.append(Layer(1, 0, Layer_Type.OUTPUT_LAYER))

        #train NN
        self.fit(X_train, y_train, max_iterations)

    def print(self):
        for layer in self.layers:
            layer.print()

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def calculate_node_value(current_layer, next_layer_node_index):
        output = 0;
        for node in current_layer.nodes:
            output += node.get_value() * node.get_weight(next_layer_node_index - 1)

        return NN.sigmoid(output)

    def forward_pass(self):
        for index in range(0, self.nof_hiden_layers):
            current_layer = self.layers[index]
            next_layer = self.layers[index + 1]

            for node_index in range (1, next_layer.nof_nodes + 1):
                next_layer.set_node_value(node_index, NN.calculate_node_value(current_layer, node_index))

        return NN.calculate_node_value(self.layers[-1], 1)

    def backward_pass(self):
        return

    def is_expected_value_match(expected_output, current_output):
        if expected_output == 1:
            if current_output > 0.8:
                return True
            else:
                return False
        else:
            if current_output < 0.2:
                return True
            else:
                return False

    def train_example(self, features, expected_output, max_iterations):
        #print("Train example")
        nof_features = len(features)
        #set input layer
        for index_feature in range (0, nof_features):
            self.layers[0].nodes[index_feature + 1].set_value(features[index_feature])

        for iteration in range (0, max_iterations):
            current_output = self.forward_pass()
            #print (current_output)

            if NN.is_expected_value_match(expected_output, current_output):
                break
            else:
                self.backward_pass()
        return

    def fit(self, X_train, y_train, max_iterations):
        nof_training_eg = np.shape(X_train)[0]

        for index in range(0, nof_training_eg):
            self.train_example(X_train[index], y[index], max_iterations)


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

myNN = NN(X_train, y_train, 5, 5)
myNN.print()
