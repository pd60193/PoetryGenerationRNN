import numpy as np
import theano
import theano.tensor as T
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../util'))
from util_parity import init_weight,parity_pairs
from sklearn.utils import shuffle

class HiddenLayer:
    def __init__(self,M1,M2,id):
        W_init,b_init = init_weight(M1,M2)
        self.W = theano.shared(W_init,'W_'+str(id))
        self.id = id
        self.M1 = M1
        self.M2 = M2
        self.b = theano.shared(b_init,'b_'+str(self.id))
        self.params = [self.W,self.b]
    def forward(self,X):
        return T.tanh(X.dot(self.W)+self.b)

class ANN(object):
    def __init__(self,hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self,X,Y,learning_rate= np.float32(0.0001),mu = np.float32(0.999),batch_sz = 50,print_period = 20,epochs = 500):
        Y = Y.astype(np.int32)
        N,D = X.shape
        K = len(set(Y))
        print(str(N)+" "+str(D)+" "+str(K))
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1,M2,count)
            self.hidden_layers.append(h)
            M1 = M2
            count+=1
        W,b = init_weight(M1,K)
        self.W = theano.shared(W.astype(np.float32),'W_logreg')
        self.b = theano.shared(b.astype(np.float32),'b_logreg')
        self.params = [self.W,self.b]
        for h in self.hidden_layers:
            self.params+=h.params
        dparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in self.params]
        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)
        #reg = np.float32(0.00001)
        learning_rate = np.float32(learning_rate)
        #rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]),thY]))# +rcost
        prediction = self.predict(thX)
        grads = T.grad(cost,self.params)
        updates = [
            (p,p+mu*dp-learning_rate*g) for p, dp, g in zip(self.params,dparams,grads)
        ]+[
          (dp,mu*dp- learning_rate*g) for dp, g in zip(dparams,grads)
        ]
        train_op = theano.function(
            inputs = [thX,thY],
            outputs =[cost,prediction],
            updates=updates,
        )
        n_batches = int(N/batch_sz)
        for x in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                XBatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                YBatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]
                c,p = train_op(XBatch,YBatch)
                if j%100 == 0:
                    #print(str(Y))
                    e = np.mean(YBatch!=p)
                    print("Cost : "+str(c)+" Error Rate: "+str(e)+" i="+str(x))

    def forward(self,X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W)+self.b)
    def predict(self,X):
        pY = self.forward(X)
        return T.argmax(pY, 1)

def wide():
    X,Y = parity_pairs(12)
    model = ANN([2048])
    model.fit(X,Y)
    
def deep():
    X,Y = parity_pairs(6)
    model = ANN([256,256,256,256,256,256,256,256])
    model.fit(X,Y)

deep()
