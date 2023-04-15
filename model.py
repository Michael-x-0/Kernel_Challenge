# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from dataviz import plot_PCA
from helper import build_laplacian
from kernel_model import KernelRigdeRegression
from kernel_model import sample_weight

class classifier:
    
    def __init__(self, model_name = "svm",ensemble = False, model_list = None,class_weight = None, w=[1.0,1.0],tr = 0,tr2 = 0,tr3 = [0.5,0.5],**kwargs):
        self.model_name = model_name
        self.ensemble = ensemble
        self.class_weight = class_weight
        self.w = w
        self.tr = tr
        self.tr2 = tr2
        self.tr3 = tr3
        if ensemble:
            assert model_list is not None
            self.model_list = model_list
            if model_name == 'svm':
                self.model = SVC(**kwargs)
            elif model_name == 'ridge':
                self.model = None #KernelRidge(kernel='linear',**kwargs)
            elif model_name == 'ridge2':
                self.model = KernelRigdeRegression(kernel='linear',class_weight=class_weight,**kwargs)
            return
        if model_name == 'svm':
            self.model = None #SVC(kernel='precomputed',class_weight=class_weight,**kwargs)
        elif model_name =='ridge':
            self.model =  None #KernelRidge(kernel='precomputed',**kwargs)
        elif model_name == 'ridge2':
            self.model = KernelRigdeRegression(kernel='precomputed',class_weight=class_weight,**kwargs)
    

    def ensemble_kernel(self,K,label = None, retrain = False):
        assert len(K) == len(self.model_list)
        pred = []
        for i,model in enumerate(self.model_list):
            if retrain:
                assert label is not None
                model.train(K[i],label)
            pred.append(model.decision_fuction(K[i]).reshape(-1,1))
        pred = np.hstack(pred)
        return pred
    
    def train(self,K,label):
        if self.ensemble:
            K = self.ensemble_kernel(K,label,retrain=True)
        if self.class_weight == 'balanced' and self.model_name == 'ridge':
            W = sample_weight(label, self.w)
            self.model.fit(K, label,W)
        else:
            self.model.fit(K,label)

    def predict(self,K, verbose = False):
        if self.ensemble:
            K = self.ensemble_kernel(K)
        pred = self.model.predict(K)
        if verbose:
            print("nb label_0 {} nb label_1 {}".format(np.sum(pred ==0),np.sum(pred == 1)))
        return pred

    def decision_fuction(self,K):
        if self.ensemble:
            K = self.ensemble_kernel(K)
        if self.model_name == 'ridge':
            pred = self.model.predict(K)
            pred[pred < self.tr] = 0
            pred[pred > 1 - self.tr2] = 1
            pred[(self.tr3[0] < pred) & (pred < self.tr3[1])] = 0.5
        else:
            pred = self.model.decision_function(K)
            pred = (pred+1)/2


        return pred
    
    def auc_score(self,K, label):
        pred = self.decision_fuction(K)
        score =None # roc_auc_score(label, pred)
        return score

    def accuracy(self, K,label):
        pred = self.predict(K)
        acc = np.sum(pred = label)/len(label)
        return acc

    def cross_val(self, K, label, cv, ratio = 0.8, balanced = False, semi = False, **params_semi):
        kf = None #KFold(n_splits=cv)
        initial_model = self.model

        if  balanced:
            index0 = kf.split(label[label == 0])
            index1 = kf.split(label[label == 1])
            score = []
            for i,x in enumerate(zip(index0,index1)):
                print("cv {}".format(i))
                train_idx = np.hstack([x[0][0],x[1][0]])
                val_idx =  np.hstack([x[1][0],x[1][1]])
                if self.ensemble:
                    K_train = [K_[train_idx,:][:,train_idx] for K_ in K]
                    K_val = [K_[val_idx,:][:,train_idx] for K_ in K]
                else:
                    K_train = K[train_idx,:][:,train_idx]
                    K_val = K[val_idx,:][:,train_idx]
                self.train(K_train, label[train_idx])
                score.append(self.auc_score(K_val, label[val_idx]))
        else:
            index= kf.split(label)
            score  =[]
            for i, x in enumerate(index):
                print("cv {}".format(i))
                train_idx = x[0]
                val_idx = x[1]
                if self.ensemble:
                    K_train = [K_[train_idx,:][:,train_idx] for K_ in K]
                    K_val = [K_[val_idx,:][:,train_idx] for K_ in K]
                else:
                    K_train = K[train_idx,:][:,train_idx]
                    K_val = K[val_idx,:][:,train_idx]
                self.train(K_train, label[train_idx])
                
                if semi:
                    lab = label[train_idx]+1
                    lab = np.hstack([lab,np.zeros(len(val_idx))])
                    idx = np.hstack([train_idx,val_idx])
                    K_ = K[idx,:][:,idx]
                    L = build_laplacian(K_)
                    Q = L+ np.eye(L.shape[0])
                    l,f = self.compute_hfs(L,lab,**params_semi)
                    pred = f[:,1]
                    score_ = None #roc_auc_score(label[val_idx],pred[-len(val_idx):])
                    score.append(score_)
                    #print("semi_supervised score {}".format(score_))
                self.train(K_train,label[train_idx])
                score.append(self.auc_score(K_val, label[val_idx]))
        self.model = initial_model
        print(f"score = {np.mean(score)}")
        return score

    def submit(self,K,name):
        if self.ensemble:
            K = self.ensemble_kernel(K)
        pred = self.decision_fuction(K)
        df = pd.DataFrame({'Id':np.arange(1,K.shape[0]+1),'Predicted':pred})
        df.to_csv(name,index = None)
    
    def compute_hfs(self,L, Y, soft=False, **params):
        
        num_samples = L.shape[0]
        Cl = np.unique(Y)
        num_classes = len(Cl)-1

        l_idx = np.arange(num_samples)[Y != 0]
        u_idx = np.arange(num_samples)[Y == 0]
        y = np.zeros((num_samples,num_classes))
        

        y[l_idx,(Y[l_idx]-1).astype(int)] = 1
        

        if not soft:    
            
            f_l = y[l_idx]
            f_u = -np.linalg.inv(L[u_idx,:][:,u_idx]).dot(L[u_idx,:][:,l_idx].dot(f_l))
            f = np.zeros_like(y)
            f[l_idx] = f_l
            f[u_idx] = f_u
        
            # ...

        else:

            C = np.zeros(num_samples)
            C[l_idx] = params['c_l']
            C[u_idx] = params['c_u']
            C = np.diag(C)
            f = np.linalg.inv((np.linalg.inv(C)@L + np.identity(num_samples)))@y
    
        
        labels = np.argmax(f, axis=1).reshape(-1,1)+1 

        return np.squeeze(labels), f

        
                
                        
        
                


