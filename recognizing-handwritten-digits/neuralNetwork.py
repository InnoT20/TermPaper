import numpy
import scipy.special
import csv


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # Матрицы весов
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Logistic (sigmoid) function
        self.activation_function = lambda x: scipy.special.expit(x)

        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        self.count = 0

    def train(self, inputs_list, targets_list):
        # преобразуем в 2хмерный массив [[...]] и транспонируем
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # для скрытого слоя перемножим веса и входные значени я
        hidden_inputs = numpy.dot(self.wih, inputs)  # wih=100x784  inputs.T=784x1  wih*inputs = 100x1
        # применем сигмоиду
        hidden_outputs = self.activation_function(hidden_inputs)

        # произведение выхода со скрытого слоя на веса
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # ошибка выходного слоя
        output_errors = targets - final_outputs  # epsilon

        # ошибка скрытого слоя
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        self.count += 1

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def backquery(self, targets_list):
        final_outputs = numpy.array(targets_list, ndmin=2).T

        final_inputs = self.inverse_activation_function(final_outputs)

        hidden_outputs = numpy.dot(self.who.T, final_inputs)

        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        inputs = numpy.dot(self.wih.T, hidden_inputs)

        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs
