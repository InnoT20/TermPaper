import numpy


def load_training(output_nodes):
    training_data_file = open("dataset/mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for record in training_data_list:
        all_values = record.split(',')

        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        target_key = int(all_values[0])

        targets = numpy.zeros(output_nodes) + 0.01
        targets[target_key] = 0.99

        yield inputs, targets


def load_test():
    test_data_file = open("dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])

        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        yield inputs, correct_label
