import numpy
from numpy import loadtxt
from sklearn import linear_model
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

def sigmoid(X):
    return 1 / (1 + numpy.exp(- X))

def cost(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta)) # predicted probability of label 1
    log_l = (-y)*numpy.log(p_1) - (1-y)*numpy.log(1-p_1) # log-likelihood vector

    return log_l.mean()

def grad(theta, X, y):
    p_1 = sigmoid(numpy.dot(X, theta))
    error = p_1 - y # difference between label and prediction
    grad = numpy.dot(error, X) / y.size # gradient vector

    return grad


def compute_cost(theta,X,y): #computes cost given predicted and actual values
    m = X.shape[0] #number of training examples
    theta = numpy.reshape(theta,(len(theta),1))

    #y = reshape(y,(len(y),1))
    
    J = (1./m) * (-numpy.transpose(y).dot(numpy.log10(sigmoid(X.dot(theta)))) - numpy.transpose(1-y).dot(numpy.log10(1-sigmoid(X.dot(theta)))))
    
    grad = numpy.transpose((1./m)*numpy.transpose(sigmoid(X.dot(theta)) - y).dot(X))
    #optimize.fmin expects a single value, so cannot return grad
    return J[0]#,grad


def compute_grad(theta, X, y):

    p_1 = sigmoid(numpy.dot(X, theta))
    error = p_1 - y # difference between label and prediction
    grad = numpy.dot(error, X) / y.size # gradient vector

    return grad



def predict(theta, X,y,intercept):
    h = X.dot(theta.T)+intercept
    m = h.size-1
    accuracy = 0.0
    while m > -1:
        if h[m] > 0.0:
            predict = 1.0
        else:
            predict = 0.0
        if predict == y[m]:
            accuracy = accuracy+1.0
        m = m - 1
    return accuracy
    
        
        
        



#load the dataset




data = loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]




# prefix an extra column of ones to the feature matrix (for intercept term)

"""

X_1 = numpy.append( numpy.ones((X.shape[0], 1)), X, axis=1)


message = "Desired error not necessarily achieved due to precision loss."

while message == "Desired error not necessarily achieved due to precision loss.":
    theta = 1.0* numpy.random.randn(3)
    theta_1 = opt.fmin_bfgs(compute_cost, theta, fprime=compute_grad,args=(X_1, y))
    message = theta_1[1]
    print "_______________________________"

print message 
print theta




#Compute accuracy on our training set
print predict(theta, X_1,y)
"""

clf = linear_model.LogisticRegression(C=1e6,solver='liblinear')
clf.fit(X, y)

predictions = clf.predict(X)
print accuracy_score(y, predictions, normalize=True)*100.0

theta = clf.coef_[0]
intercept =  clf.intercept_
print predict(theta, X, y,intercept)


print "done"
