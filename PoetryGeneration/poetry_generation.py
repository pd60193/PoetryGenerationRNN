import numpy as np
import theano
import theano.tensor as T
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'../util'))
from sklearn.utils import shuffle
from util_parity import init_weight,get_robert_frost

class SimpleRNN(object):
    def __init__(self,D,M,V):
        self.D = D
        self.M = M
        self.V = V

    def set(self,We,Wx,h0,Wh,bh,Wo,bo,lr):
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.Wo = theano.shared(Wo)
        self.h0 = theano.shared(h0)
        self.bh = theano.shared(bh)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.h0, self.Wh, self.bh, self.Wo, self.bo]
        thX = T.ivector('X')
        Ei = self.We[thX]
        thY = T.ivector('Y')
        def recurrence(x_t,h_t1):
            h_t = self.f(x_t.dot(self.Wx)+h_t1.dot(self.Wh)+self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo)+self.bo)
            return h_t,y_t
        [h,y],_ = theano.scan(
            fn = recurrence,
            outputs_info = [self.h0,None],
            sequences = Ei,
            n_steps = Ei.shape[0]
            )
        py_x = y[:,0,:]
        prediction = T.argmax(py_x,axis = 1)
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]),thY]))
        grads = T.grad(cost,self.params)
        updates= [
            (p,p-lr*g) for p,g in zip(self.params,grads)
            ]
        self.train_op = theano.function(inputs = [thX,thY],outputs = [prediction,cost],updates = updates)
        self.predict_op = theano.function(inputs = [thX],outputs=prediction)



    def fit(self,X,lr = 10e-1,mu = 0.99,activation = T.tanh,epochs = 500):
        N = len(X) 
        V = self.V
        D = self.D
        M = self.M
        self.f = activation
        We = np.random.randn(V,D)/np.sqrt(V+D)
        We = We.astype(np.float32)
        Wx,h0 = init_weight(D,M)
        Wh,bh = init_weight(M,M)
        Wo,bo = init_weight(M,V)
        self.set(We,Wx,h0,Wh,bh,Wo,bo,lr)
        n_total =sum((len(sentence)+1) for sentence in X)
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            cost = 0
            for j in range(N):
                input_seq = [0]+X[j]
                output_seq = X[j]+[1]
                p,c = self.train_op(input_seq,output_seq)
                cost+=c
                for k in range(len(p)):
                    if(p[k]==output_seq[k]):
                        n_correct +=1
            crate = float(n_correct)/n_total
            print("i: "+str(i)+" cost "+str(cost)+" crate: "+str(crate))

    def generate(self,pi,word2idx):
        idx2word = {v:k for k,v in word2idx.items()}
        V = len(pi)
        n_lines = 0
        X = [np.random.choice(V,p=pi)]
        print(idx2word[X[0]],end=' ')
        while n_lines<4:
            p = self.predict_op(X)[-1]
            X+=[p]
            if p>1:
                word = idx2word[p]
                print(word,end=' ')
            elif p==1:
                n_lines+=1
                print('')
                if n_lines <4:
                    X = [np.random.choice(V,p=pi)]
                    print(idx2word[X[0]],end=' ')


def generate_poetry():
    sentences,word2idx = get_robert_frost("../util/robert_frost.txt")
    rnn = SimpleRNN(30,300,len(word2idx))
    rnn.fit(sentences,lr = np.float32(10e-4),activation = T.nnet.relu,epochs=2000)
    V = len(word2idx)
    pi = np.zeros(V)
    for sentence in sentences:
        pi[sentence[0]]+=1
    pi = pi/pi.sum()
    rnn.generate(pi,word2idx)

generate_poetry()
