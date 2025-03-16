
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    means = np.array([X[y.flatten() == i].mean(axis=0) for i in np.unique(y)])
    covmat = np.cov(X.T)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    means = np.array([X[y.flatten() == i].mean(axis=0) for i in np.unique(y)])
    covmats = [np.cov(X[y.flatten() == i].T) for i in np.unique(y)]
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    inv_covmat = np.linalg.inv(covmat)
    preds = []
    for x in Xtest:
        probs = [np.dot(np.dot((x - mean).T, inv_covmat), (x - mean)) for mean in means]
        preds.append(np.argmin(probs) + 1)
    acc = np.mean(preds == ytest.flatten())
    ypred = np.array(preds)
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    preds = []
    for x in Xtest:
        probs = [np.dot(np.dot((x - mean).T, np.linalg.inv(covmat)), (x - mean))
                 for mean, covmat in zip(means, covmats)]
        preds.append(np.argmin(probs) + 1)
    acc = np.mean(preds == ytest.flatten())
    ypred = np.array(preds)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    d = X.shape[1]
    w = np.linalg.inv(X.T @ X + lambd * np.eye(d)) @ X.T @ y
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD
    predictions = Xtest @ w
    mse = np.mean((ytest - predictions) ** 2)
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD
    w = w.reshape((-1,1))
    error = 0.5 * (np.sum((y - X @ w) ** 2) + 0.5 * lambd * np.sum(w ** 2))
    error_grad = -(X.T @ (y - X @ w)) + lambd * w
    return error, error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))

    # IMPLEMENT THIS METHOD
    Xp = np.ones((x.shape[0], p+1))
    for i in range(1,p+1):
        Xp[:,i] = x**i
    return Xp

"""# Main Script

"""

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open(r"sample.pickle",'rb'))
else:
    X,y,Xtest,ytest = pickle.load(open(r"sample.pickle",'rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()

if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open(r"diabetes.pickle", 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open(r"diabetes.pickle", 'rb'), encoding='latin1')

X_i = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)

w_no_intercept = learnOLERegression(X, y)
mse_train_no_intercept = testOLERegression(w_no_intercept, X, y)
mse_test_no_intercept = testOLERegression(w_no_intercept, Xtest, ytest)

w_with_intercept = learnOLERegression(X_i, y)
mse_train_with_intercept = testOLERegression(w_with_intercept, X_i, y)
mse_test_with_intercept = testOLERegression(w_with_intercept, Xtest_i, ytest)

print(f"MSE without intercept (Training Data): {mse_train_no_intercept:.4f}")
print(f"MSE without intercept (Test Data): {mse_test_no_intercept:.4f}")
print(f"MSE with intercept (Training Data): {mse_train_with_intercept:.4f}")
print(f"MSE with intercept (Test Data): {mse_test_with_intercept:.4f}")

# Calculate and store the MSE for Train and Test Data
k = 101
lambdas = np.linspace(0, 1, num=k)
mses3_train = np.zeros((k, 1))
mses3 = np.zeros((k, 1))
wgts = []

for i, lambd in enumerate(lambdas):
    w_l = learnRidgeRegression(X_i, y, lambd)
    mses3_train[i] = testOLERegression(w_l, X_i, y)
    mses3[i] = testOLERegression(w_l, Xtest_i, ytest)
    wgts.append(w_l)


average_mse_train = np.mean(mses3_train)
average_mse_test = np.mean(mses3)

# Print the results
print(f"Average MSE for Train Data: {average_mse_train:.4f}")
print(f"Average MSE for Test Data: {average_mse_test:.4f}")

w_n = learnOLERegression(X_i, y)
print("Magnitude of Weights without Regularization:")
print(np.linalg.norm(w_n, ord=2))

print("Magnitude of Weights with Regularization:")
for i, w in enumerate(wgts):
    print(f"Lambda = {lambdas[i]:.4f}: {np.linalg.norm(w, ord=2):.4f}")

# Plotting
fig = plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(lambdas, mses3_train, label='Train MSE')
plt.title('MSE for Train Data')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lambdas, mses3, label='Test MSE')
plt.title('MSE for Test Data')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.legend()

plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init.flatten(), jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
lambda_train = lambdas[np.argmin(mses4_train)]
lambda_test = lambdas[np.argmin(mses4)]
print('Optimal lambda based on train data: ' + str(lambda_train))
print('Optimal lambda based on test data: ' + str(lambda_test))

mse_tr_min = np.min(mses4_train)
mse_min = np.min(mses4)

print(f"Minimum MSE for Train Data: {mse_tr_min:.4f}")
print(f"Minimum MSE for Test Data: {mse_min:.4f}")

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

