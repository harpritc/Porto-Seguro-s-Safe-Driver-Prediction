import random
from enum import Enum
from macpath import norm_error
import numpy as np
import math
import sys
import pandas as pd

class Layer_Type(Enum):
    INPUT_LAYER = 1
    HIDDEN_LAYER = 2
    OUTPUT_LAYER = 3

class Node:
    def __init__(self, number_of_weights, value=0,delta=0):
        self.weights = []
        self.value = value
        self.delta = delta
        for index in range(0, number_of_weights):
            self.weights.append(random.random())

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_delta(self, delta):
        self.delta = delta

    def get_delta(self):
        return self.delta

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

        # include extra node for bias in each layer
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

    def set_node_delta(self, index, delta):
        self.nodes[index].set_delta(delta)

    def print(self):
        print("Number of nodes: %d" % (self.nof_nodes))
        for node in self.nodes:
            print("node value: %f" % (node.value))
            print("delta value: %f" % (node.delta))
            for weight in node.weights:
                print(weight)



    def print(self):
        for index in range (1, self.nof_nodes + 1):
            print('\tNeuron{} weights: {}'.format(index, self.nodes[index].weights))
            #print("node value: %f" % (node.value))
            #for weight in node.weights:
            #    print(weight)

class NN:
    def __init__(self, X_train, y_train, max_iterations, nof_nodes):
        self.nof_hiden_layers = len(nof_nodes)
        self.layers = []

        #input layer
        self.layers.append(Layer(np.shape(X_train)[1], nof_nodes[0], Layer_Type.INPUT_LAYER))

        #hidden layers
        for index in range(0, self.nof_hiden_layers - 1):
            self.layers.append(Layer(nof_nodes[index], nof_nodes[index+1], Layer_Type.HIDDEN_LAYER))

        self.layers.append(Layer(nof_nodes[self.nof_hiden_layers - 1], 1, Layer_Type.HIDDEN_LAYER))

        #output layer
        self.layers.append(Layer(1, 0, Layer_Type.OUTPUT_LAYER))

        #train NN
        self.fit(X_train, y_train, max_iterations)

    def print(self):
        for index in range (0, self.nof_hiden_layers + 1):
            print('Layer {}'.format(index))
            self.layers[index].print()

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

     def calculate_delta_value(next_layer, node):
        output = 0;
        for index in range(1,next_layer.nof_nodes+1):
            output += next_layer.nodes[index].get_delta() * node.get_weight(index - 1)
        delta = (node.value) * (1-node.value) * output
        return delta

    def weightUpdate_value(next_layer, node):
        for index in range(1,next_layer.nof_nodes+1):
            newWeight = node.get_weight(index - 1) + ( next_layer.nodes[index].get_delta() * node.value )
            node.set_weight(index - 1, newWeight)
            print("New Weight %f" % (newWeight))
	
    def calculate_node_value(current_layer, next_layer_node_index):
        output = 0;
        for node in current_layer.nodes:
            output += node.get_value() * node.get_weight(next_layer_node_index - 1)

        return NN.sigmoid(output)

    def forward_pass(self):
        for index in range(0, self.nof_hiden_layers + 1):
            current_layer = self.layers[index]
            next_layer = self.layers[index + 1]

            for node_index in range (1, next_layer.nof_nodes + 1):
                next_layer.set_node_value(node_index, NN.calculate_node_value(current_layer, node_index))

        return self.layers[-1].nodes[1].value

    def backward_pass(self,current_output,expected_output):
        self.layers[-1].set_node_delta(1,current_output*(1-current_output)*(expected_output-current_output));

        for index in range(self.nof_hiden_layers,-1,-1):
            current_layer = self.layers[index]
            next_layer = self.layers[index + 1]

            for node_index in range(1, current_layer.nof_nodes + 1):
                current_layer.set_node_delta(node_index, NN.calculate_delta_value(next_layer, current_layer.nodes[node_index]))
            for node_index in range(1, current_layer.nof_nodes + 1):
                NN.weightUpdate_value(next_layer, current_layer.nodes[node_index])


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
		self.backward_pass(current_output,expected_output)

        return

    def fit(self, X_train, y_train, max_iterations):
        nof_training_eg = np.shape(X_train)[0]

        for index in range(0, nof_training_eg):
            self.train_example(X_train[index], y_train[index], max_iterations)

    def get_prediction(self, features):
        #set input layer
        for index_feature in range (0, len(features)):
            self.layers[0].nodes[index_feature + 1].set_value(features[index_feature])

        output = self.forward_pass()

        if output >= 0.6:
            return 1
        else:
            return 0

def train_test_split(input_array, prediction_array, training_percent):
    if len(input_array) != len(prediction_array):
        raise Exception('length of input and prediction does not match')

    length_array = len(input_array)
    list_random = random.sample(range(0, length_array), int(length_array * (100 - training_percent) / 100))
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

def main(argv):
    if len(argv) < 5 :
        print('Usage: ANN.py <DataSetFile> <TrainingPercent> <Maximum_iterations> <number_of_hidden_layers> [Hidden_layer_sizes]')
        print('Example: ANN.py ds1 80 200 2 4 2')
        sys.exit()

    X = [[0, 0], [0, 1], [1, 0], [1, 1], [1,1]] * 2
    y = [0, 1, 1, 0, 1] * 2
    #X_train, X_test, y_train, y_test = train_test_split(X, y, 20)

    data_set = argv[0]
    training_percent = int(argv[1])
    max_iterations = int(argv[2])
    nof_hiden_layer = int(argv[3])
    nof_nodes_in_hidden_layer = []

    for index in range (0, nof_hiden_layer):
        nof_nodes_in_hidden_layer.append(int(argv[4+index]))
	
    df = pd.read_csv(data_set)
    y = df['Class']
    del df['Class']
    X = df.as_matrix()

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, training_percent)

    print(X_train)
    print(X_test)

    print(y_train)
    print(y_test)

    #Create Neural Network and train
    myNN = NN(X_train, y_train, nof_hiden_layer, nof_nodes_in_hidden_layer)

    #print ANN
    myNN.print()

    #Training Accuracy
    countErrors = 0
    for index in range (0, len(y_train)):
        prediction = myNN.get_prediction(X_train[index])
        if prediction != y_train[index]:
            countErrors += 1
    print('Total training error = {}'.format(countErrors * 100 / len(y_train)))

    #Test Accuracy
    countErrors = 0
    for index in range (0, len(y_test)):
        prediction = myNN.get_prediction(X_test[index])
        if prediction != y_test[index]:
            countErrors += 1
    print('Total test error = {}'.format(countErrors * 100 / len(y_test)))

if __name__ == "__main__":
	main(sys.argv[1:])
