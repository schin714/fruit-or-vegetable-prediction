import numpy as np
from random import choice
import matplotlib.pyplot as plt
import csv

def predict_y(model,data):
    '''
    Make the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: predictions
    '''
    result = np.matmul(data,model)
    return np.sign(result)

def get_input():
	'''
    Get the input for a fruit or vegetable in 100g serving from user
    :return: Array of inputs
    '''
	print("Input calorie and carb content of 100g serving of a fruit or vegetable:")
	input_string = input()
	input_arr = np.array([float(x) for x in input_string.split(',')])
	input_arr = np.append(input_arr, 1)
	return input_arr

def predict_input(model,input):
	'''
    Make the prediction if it's a fruit or vegetable from the input data
    :param model: model vector
    :param input:  input data
    :return: predictions
    '''
	result = np.matmul(input,model)
	result = 'Fruit' if np.sign(result) == 1 else 'Vegetable'

	print("Result = ", result)

def train_perceptron(training_data, old_model):
    '''
    Train a perceptron model given a set of training data
    :param training_data: A list of data points, where training_data[0]
    contains the data points and training_data[1] contains the labels.
    Labels are +1/-1.
    :return: learned model vector
    '''
    X = training_data[0]
    y = training_data[1]
    w = old_model
    iteration = 1
    while True:
        # compute results according to the hypothesis
        pr_y = predict_y(w,X)
        mc_x = []
        mc_y = 0

        # get incorrect predictions (you can get the indices) and pick a misclassified example
        for i in range(y.size):
            if(pr_y[i] != y[i]):
                mc_x = X[i]
                mc_y = pr_y[i] * -1
                break

        # Check the convergence criteria (if there are no misclassified
        # points, the PLA is converged and we can stop.)
        if(mc_y == 0):
            break

        # Update the weight vector with perceptron update rule
        new_x = mc_x * mc_y * 0.01
        w = np.add(w, new_x)

        iteration += 1

        if(iteration >= 1000):
            break

    return w

def print_prediction(data,predictions):
    '''
    Print the predictions given the dataset and the learned model.
    :param data:  data points
    :param predictions: predictions
    :return: nothing
    '''
    for i in range(len(data)):
        print("{}: -> {}".format(data[i], predictions[i]))

def graph_data(data_vals, model):
    '''
    Graph the data points(labeled) and the line created by the model
    :param data_vals:  data points and class together
    :param model: trained model
    :return: nothing
    '''
    #Create array of colors to use for the points
    data_size = data_vals[1].size
    colors = ['yellow'] * data_size
    for i in range(data_size):
        if(data_vals[1][i] == 1):
            colors[i] = 'red'
        else:
            colors[i] = 'blue'

    #Graph data points and lines
    spacers = np.linspace(0,100,100)
    line = (-1 * model[0]/model[1])*spacers + (-1 * model[2]/model[1])
    # plt.ylim([-0.2,1.2])
    # plt.xlim([-0.2,1.2])
    plt.scatter(data_vals[0].T[0], data_vals[0].T[1], s=30, color=colors)
    plt.plot(spacers, line, '-b', label='Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Classification')
    plt.show()

if __name__ == '__main__':

	#get data and classifications from a csv file
	fv_data = []
	fv_class = []
	with open('fruitsvegetables.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count != 0:
				fv_data.append([float(row[1]),float(row[2]),1])
				row_class = 1 if (row[3] == 'yes') else -1
				fv_class.append(row_class)

			line_count += 1

	#convert data and class arrays to np arrays and bundle into array of the values
	fv_data = np.array(fv_data)
	fv_class = np.array(fv_class)
	fv_vals = [fv_data,fv_class]

	#get the old trained model from a trained_model.txt file or create a random model if there isn't already one
	model_size = fv_data[1].size
	old_model = np.random.rand(model_size) #np.zeros(model_size) or np.random.rand(model_size)
	with open('trained_model.txt') as model_file:
		content = model_file.read()
		old_model = np.array([float(w) for w in content.split(',')]) # np.array([0, 3, 5])

	trained_model = train_perceptron(fv_vals, old_model)

	#write trained_model to a file
	with open('trained_model.txt','w+') as model_file:
		model_str = ','.join(str(w) for w in trained_model) # '0,3,5'
		model_file.write(model_str)

	#print model and print predictions
	print("Model:", trained_model)
	predictions = predict_y(trained_model, fv_data)
	print_prediction(fv_data, predictions)

	input_arr = get_input()
	predict_input(trained_model, input_arr)

	#graph data last since the rest doesn't run until window is closed
	graph_data(fv_vals,trained_model)
