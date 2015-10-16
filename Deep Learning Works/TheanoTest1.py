import theano 
from theano import tensor as T 
import numpy as np 


trainingX = np.linspace(-1, 1, 101)
trainingy = 2*trainingX+np.random.randn(*trainingX.shape) * 0.33 

X = T.scalar()
Y = T.scalar()

def model(X,w):
    return X*w 

w = theano.shared(np.asarray(0., dtype=float))
y = model(X, w)

cost = T.mean(T.sqr(y-Y))
gradient = T.grad(cost=cost,wrt=w)
update = [[w,w-gradient * 0.01]]


train = theano.function(inputs = [X,Y], outputs = cost, updates=update,allow_input_downcast=True)


for i in range(100):
    for x,y in zip(trainingX,trainingy):
        output = train(x,y)
        print output
        print w.get_value()
        print "______________"




print trainingX

print "done"