import numpy as np
import theano
import theano.tensor as T
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../util'))
from util_parity import init_weight,parity_pairs_with_labels
from sklearn.utils import shuffle


class SimpleRNN(object):
    def __init__(self,M):
        self.M = M
    def fit(self,X,Y,lr=np.float32(0.0001),epochs = 500,mu = np.float32(0.99),activation=T.tanh):
        D = X[0].shape[1]
        K = len(set(Y.flatten()))
        N = len(Y)
        self.f = activation
        Wh,h0 = init_weight(self.M,self.M)
        Wx,bx = init_weight(D,self.M)
        Wy,by = init_weight(self.M,K)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.Wy = theano.shared(Wy)
        self.h0 = theano.shared(h0)
        self.bx = theano.shared(bx)
        self.by = theano.shared(by)
        self.params = [self.Wx,self.bx,self.Wh,self.h0,self.Wy,self.by]
        thX = T.fmatrix('X')
        thY = T.ivector('Y')
        def recurrence(x_t,h_t1):
            h_t = self.f(h_t1.dot(self.Wh)+x_t.dot(self.Wx)+self.bx)
            y_t = T.nnet.softmax(h_t.dot(self.Wy)+self.by)
            return h_t,y_t
        [h,y],_ = theano.scan(
            fn = recurrence,
            outputs_info = [self.h0,None],
            sequences=thX,
            n_steps = thX.shape[0],
        )
        #print(str(y.eval()))
        py_x = y[:,0,:]
        prediction = T.argmax(py_x,axis = 1)
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]),thY]))
        
        grads = T.grad(cost,self.params)
        dparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in self.params]
        updates = [
            	(p,p+mu*dp- lr*g) for p,dp,g in zip(self.params,dparams,grads)
        ] + [
            (dp,mu*dp-lr*g) for dp,g in zip(dparams, grads)
        ]
        self.predict_op = theano.function(inputs = [thX],outputs=[prediction])
        self.train_op = theano.function(inputs = [thX,thY],outputs = [cost,prediction,y],updates=updates)
        for i in range(epochs):
            #X,Y = shuffle(X,Y)
            n_correct = 0
            count = 0 
            for j in range(N):
                c,p,rout = self.train_op(X[j],Y[j])
                cost+=c
                if p[-1] == Y[j,-1]:
                    n_correct+=1
                count+=1
            #print("shape y:"+str(rout.shape))
            print("cost: "+str(c)+" i: "+str(i)+" crate: "+str(n_correct/count))
            if(int(n_correct/count)==1):
                string_inp = input()
                input_list = [int(a) for a in string_inp]
                output_val = self.predict_op(np.array(input_list).reshape(12,1).astype(np.float32))
                print("output val: "+str(output_val[-1]))               


X,Y = parity_pairs_with_labels(12)
rnn = SimpleRNN(4)
rnn.fit(X,Y)

