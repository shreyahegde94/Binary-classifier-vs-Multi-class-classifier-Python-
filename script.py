import numpy as np
import pickle
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    w = initialWeights
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    yi = labeli
    error = 0
    error_grad = np.zeros((n_features + 1, 1))
    # Add Bias at the start
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    #np.set_printoptions(threshold=np.nan)
    training_data_size = n_data
    feature_vector_size = n_features
    #print("Training data before " , np.shape(train_data))
    training_data_reshaped = np.reshape(train_data, (training_data_size, feature_vector_size))
    train_data_biased = np.insert(training_data_reshaped, 0, 1, axis = 1)
    w = w.reshape((n_features+1,1))
    #print("Training data after ", np.shape(train_data_biased))
    #print("Weights Shape ", np.shape(w))
    theta = sigmoid(np.dot(train_data_biased,w))
    #print("Theta Shape " , np.shape(theta))
    #print("Output classes shape ", np.shape(yi))
    error = np.sum(np.multiply(yi, np.log(theta)) + np.multiply((1-yi),np.log(1-theta)))
    error = (error) * (-1/training_data_size)  # Performs the process of -1/N
    error_grad = np.sum((np.multiply((theta - yi),train_data_biased)), axis=0)
    error_grad = error_grad / training_data_size
    error_grad = error_grad.flatten()
    #print("Error", error)
    #print("Error Grad ", error_grad)
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    n_data = data.shape[0]
    n_features = data.shape[1]
    data_reshaped = np.reshape(data,(n_data,n_features))
    #print("Training data before " , np.shape(train_data_reshaped))
    # Bias term to the input data - Added at the Beginning
    data = np.insert(data_reshaped, 0, 1, axis = 1)
    #data=np.hstack((np.ones((data.shape[0],1)),data))
    y = sigmoid(np.dot(data,W))
    #print("Shape of y ", np.shape(y))
    label = np.argmax(y,axis = 1)
    label = np.array([label]).reshape(label.shape[0],1)
    #print("Label Returned " , np.shape(label))
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    training_data_size = n_data
    feature_vector_size = n_feature

    training_data_reshaped = np.reshape(train_data, (training_data_size, feature_vector_size))
    train_data_biased = np.insert(training_data_reshaped, 0, 1, axis = 1)
    #train_data_biased = np.hstack((np.ones((train_data.shape[0], 1)),train_data))
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    w = params
    #print("Before Shape W", np.shape(W))
    w = w.reshape((n_feature + 1,n_class))
    #print("After Shape W",  np.s hape(w))
    exp_wx = np.exp(np.dot(train_data_biased, w))
    theta_nk = exp_wx/exp_wx.sum(axis=1)[:,None]
    #theta_nk = np.divide(exp_wx, sum_wx)
    #print(theta_nk)
    #print("Shape of theta_nk", np.shape(theta_nk))
    #print("Shape of Label i ", np.shape(labeli))
    error = np.multiply(labeli, np.log(theta_nk))
    error = np.sum(error)
    error = -1 * (error)
    error = error/n_data

    thetalabeldiff = theta_nk - labeli
    #print("Shape of Theta Label Diff ", np.shape(thetalabeldiff))
    #print("Shape of Training data ", np.shape(train_data_biased))
    error_grad = np.dot(train_data_biased.T, thetalabeldiff)
    error_grad = error_grad/n_data
    error_grad = error_grad.flatten()
    ##################-                                                                                                     n
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))
    data_reshaped = np.reshape(data, (data.shape[0], data.shape[1]))
    data_biased = np.insert(data_reshaped, 0, 1, axis = 1)
    #data_biased = np.hstack((np.ones((data.shape[0], 1)),data))
    exp_wx = np.exp(np.dot(data_biased, W))
    theta_nk = exp_wx/exp_wx.sum(axis=1)[:,None]
    label = np.argmax(theta_nk,axis=1);
    label = label.reshape((data.shape[0],1))
    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
print("Logistic Regression using BLR")
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

#Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

#Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

#Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
#Pickle Generation
f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()

#Script for Extra Credit Part

# FOR EXTRA CREDIT ONLY
print("Logistic Regression using MLR")
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

#Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# # Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# # Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
#Pickle Generation
f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()

#Script for Support Vector Machine

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
train_label = train_label.ravel()
validation_label = validation_label.ravel()
test_label = test_label.ravel()
##################
# Linear kernel
##################
print('-------------- Linear SVM-------------------\n')
model = svm.SVC(kernel='linear')
model.fit(train_data,train_label)
predicted_values = model.predict(train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_values == train_label).astype(float))) + '%')
predicted_values = model.predict(validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_values == validation_label).astype(float))) + '%')
predicted_values = model.predict(test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_values == test_label).astype(float))) + '%')

##################
# RBF kernel, Gamma = 1
##################
print('-------------- RBF, gamma =1  SVM-------------------\n')
model = svm.SVC(kernel='rbf',gamma=1)
model.fit(train_data,train_label)
predicted_values = model.predict(train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_values == train_label).astype(float))) + '%')
predicted_values = model.predict(validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_values == validation_label).astype(float))) + '%')
predicted_values = model.predict(test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_values == test_label).astype(float))) + '%')

#################
#RBF kernel, Gamma = default
#################
print('-------------- RBF, gamma = default SVM-------------------\n')
model = svm.SVC(kernel='rbf')
model.fit(train_data,train_label)
predicted_values = model.predict(train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_values == train_label).astype(float))) + '%')
predicted_values = model.predict(validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_values == validation_label).astype(float))) + '%')
predicted_values = model.predict(test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_values == test_label).astype(float))) + '%')


print('-------------- RBF, gamma = default, varying C values 10,20,...100 SVM-------------------\n')
for i in range(0,101,10):
    if(i == 0):
    	i = 1
    model = svm.SVC(C=i,kernel='rbf')
    model.fit(train_data,train_label)
    predicted_values = model.predict(train_data);
    print('C =' + str(i)+'\n')
    print('\n Training set Accuracy:' + str(100*np.mean((predicted_values == train_label).astype(float))) + '%')
    predicted_values = model.predict(validation_data);
    print('\n Validation set Accuracy:' + str(100*np.mean((predicted_values == validation_label).astype(float))) + '%')
    predicted_values = model.predict(test_data);
    print('\n Testing set Accuracy:' + str(100*np.mean((predicted_values == test_label).astype(float))) + '%')
