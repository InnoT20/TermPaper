import neuralNetwork
import numpy
import load_mnist
import matplotlib.pyplot as pl

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
scorecard = []
epochs = 2

# NN
network = neuralNetwork.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# for epoch in range(epochs):
for row in load_mnist.load_training(output_nodes):
    network.train(*row)

for row in load_mnist.load_test():
    correct_label = row[1]

    outputs = network.query(row[0])
    label = numpy.argmax(outputs)

    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)


scorecard_array = numpy.asarray(scorecard)
print("Эффективность =", scorecard_array.sum() / scorecard_array.size)
