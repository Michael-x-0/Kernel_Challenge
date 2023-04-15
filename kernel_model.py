
import numpy as np

def sample_weight(label, w=None):
    n = len(label)
    unique_label = np.unique(label)
    assert len(unique_label) == 2, "the number of labels must be equal to 2"
    idx = [np.where(label==l)[0] for l in  unique_label]
    W = np.ones(n)
    if w is  None:
        w = np.array([len(x) for x in  idx])
        w = len(label)/w
        #w = w/np.sum(w)
    else:
        assert len(w) == len(unique_label), "the number must be equal to the number of label"
   
    for i,index in enumerate(idx):
        W[index] = w[i]
    #W = np.diag(W)
    return W

class KernelRigdeRegression:
    def __init__(self,kernel,C, class_weight = None):
        self.C = C
        self.kernel = kernel
        self.class_weight = class_weight
        
    def fit(self,X,label,W=None):
        if type(label) == list:
            label = np.array(label)
        n = len(label)
        I = np.eye(n)
        if self.kernel == 'linear':
            K = X@X.T
        elif self.kernel == 'precomputed':
            K = X
        if self.class_weight == 'balanced':
            W = np.diag(sample_weight(label))
        else:
            W = np.eye(n)
        W = W**0.5
        alpha = W@np.linalg.inv(W@K@W + n*self.C*I)@W@label
        self.alpha = alpha
        
    def decision_function(self,K):
        return K.dot(self.alpha)
        
            
