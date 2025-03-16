'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt

# Do not change this
def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sep=(1+np.exp(-z))
    dhe=1/sep
    return dhe
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Convert training labels to one-hot encoding
    train_label_onehot = np.eye(n_class)[training_label.astype(int)]

    # 1. Feedforward Propagation
    # Add bias unit to training data
    training_data = np.hstack((training_data, np.ones((training_data.shape[0], 1))))

    # Hidden layer output
    a = training_data @ w1.T
    z = sigmoid(a)

    # Add bias unit to hidden layer output
    z = np.hstack((z, np.ones((z.shape[0], 1))))

    # Output layer output
    b = z @ w2.T
    o = sigmoid(b)

    # 2. Error Function Calculation (Negative Log-Likelihood with Regularization)
    # Cross-entropy error
    error = -np.mean(np.sum(train_label_onehot * np.log(o) + (1 - train_label_onehot) * np.log(1 - o), axis=1))

    # Regularization term
    regularization = (lambdaval / (2 * training_data.shape[0])) * (np.sum(w1[:, :-1]**2) + np.sum(w2[:, :-1]**2))
    obj_val = error + regularization

    # 3. Backpropagation to Calculate Gradients
    # Output layer delta
    delta_output = o - train_label_onehot

    # Hidden layer delta (no bias in backpropagation)
    delta_hidden = (delta_output @ w2[:, :-1]) * (z[:, :-1] * (1 - z[:, :-1]))

    # Calculate gradients for w1 and w2
    grad_w1 = (delta_hidden.T @ training_data) / training_data.shape[0]
    grad_w2 = (delta_output.T @ z) / training_data.shape[0]

    # Add regularization term to gradients
    grad_w1[:, :-1] += (lambdaval / training_data.shape[0]) * w1[:, :-1]
    grad_w2[:, :-1] += (lambdaval / training_data.shape[0]) * w2[:, :-1]




    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    return (obj_val, obj_grad)
    
# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    # Add bias term to the input data
    data = np.hstack((data, np.ones((data.shape[0], 1))))

    # Calculate hidden layer output
    a = data @ w1.T
    z = sigmoid(a)

    # Add bias term to the hidden layer output
    z = np.hstack((z, np.ones((z.shape[0], 1))))

    # Calculate output layer output
    b = z @ w2.T
    o = sigmoid(b)

    # Predict labels by taking the index of the maximum value in each row
    labels = o.argmax(axis=1)


    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('/content/face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')



from scipy.optimize import minimize
import numpy as np
import time

# Loop to train for different lambda values and hidden neurons
lambda_values = range(0, 65, 10)
hidden_neurons = [4, 8, 12, 16, 20]
results = []

for lambdaval in lambda_values:
    for n_hidden in hidden_neurons:
        # Initialize weights for the specific configuration
        ini_w1 = initializeWeights(n_input, n_hidden)
        ini_w2 = initializeWeights(n_hidden, n_class)
        iW = np.concatenate((ini_w1.flatten(), ini_w2.flatten()), 0)
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        opts = {'maxiter': 50}
        start_time = time.time()
        nn_params = minimize(nnObjFunction, iW, jac=True, args=args, method='CG', options=opts)
        params = nn_params.get('x')
        end_time = time.time()
        training_time = end_time - start_time
        w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
        train_accuracy = 100 * np.mean((nnPredict(w1, w2, train_data) == train_label).astype(float))
        validation_accuracy = 100 * np.mean((nnPredict(w1, w2, validation_data) == validation_label).astype(float))
        test_accuracy = 100 * np.mean((nnPredict(w1, w2, test_data) == test_label).astype(float))

        # Store the results
        results.append({'lambda': lambdaval,'hidden_neurons': n_hidden,'train_accuracy': train_accuracy,'validation_accuracy': validation_accuracy,'test_accuracy': test_accuracy,'training_time': training_time})

        print(f"\nLambda: {lambdaval}, Hidden Neurons: {n_hidden}")
        print(f"Training set Accuracy: {train_accuracy}%")
        print(f"Validation set Accuracy: {validation_accuracy}%")
        print(f"Test set Accuracy: {test_accuracy}%")

        df_results = pd.DataFrame(results)

        df_results.to_csv("facennresults.csv", index=False)

        print("Results saved to 'results.csv'")


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df=pd.read_csv('facenn_training_results.csv')
#df
d=pd.read_csv('facennresults.csv')
#df
plt.figure(figsize=(16, 8))





fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
metrics = ['train_accuracy', 'validation_accuracy', 'test_accuracy']
titles = ['Training Accuracy', 'Validation Accuracy', 'Test Accuracy']

for i, metric in enumerate(metrics):
    sns.lineplot(data=d, x='lambda', y=metric, hue='hidden_neurons', ax=ax[i])
    ax[i].set_title(f'{titles[i]} by Lambda')
    ax[i].set_xlabel('Lambda')
    ax[i].set_ylabel('Accuracy')
    ax[i].autoscale()

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for i, metric in enumerate(metrics):
    sns.lineplot(data=d, x='hidden_neurons', y=metric, ax=ax[i], errorbar='sd', marker='o')
    ax[i].set_title(f'Average {titles[i]} Accuracy by Hidden Units')
    ax[i].set_xlabel('Hidden Units')
    ax[i].set_ylabel('Accuracy')

plt.tight_layout()
plt.show()

avg_training_time = d.groupby('hidden_neurons')['training_time'].mean().reset_index()
plt.figure(figsize=(8,6))
plt.plot(avg_training_time['hidden_neurons'],avg_training_time['training_time'],color='skyblue',marker='o',label='Average Training Time')
plt.title('Average Training Time vs. Hidden Neurons')
plt.xlabel('Hidden Neurons')
plt.ylabel('Average Training Time')
plt.grid(True)
plt.tight_layout()
plt.show()