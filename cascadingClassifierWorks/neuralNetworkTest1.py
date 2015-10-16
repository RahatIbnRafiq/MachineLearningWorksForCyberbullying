import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn import datasets
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target

net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
        # layer parameters:
        input_shape=X.shape,
        hidden_num_units=100,  # number of units in 'hidden' layer
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=10,
        verbose=1,
        )


print "training"
net1.fit(X, Y)
print "predicting"
predictions = net1.predict(X)

print classification_report(Y, predictions)

print "done"