import numpy as np
from scipy import sparse
import networkx as nx
from tqdm import tqdm
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
import multiprocessing
from multiprocessing import Value, Array
from functools import partial
from grakel import RandomWalkLabeled
from grakel import Graph
from pathos.multiprocessing import ProcessingPool as Pool
import pickle
import time
import os


class RandonWalk:

    def __init__(self,dic = False):
        self.dic = False

    def good_format(self,X,dic = False):
        res = []
        for G in X:
            m = nx.adjacency_matrix(G)
            if dic:
                m = m.toarray()
                res.append(Graph(m,self.get_label_nodes(G,dic)))
            else:
                res.append((m.reshape(-1,1),m.reshape(1,-1),self.get_label_nodes(G,dic),self.get_label_edges(G,dic)))
            
        return res

    def get_label_nodes(self,G, dic = False):
        if dic:
            label = {}
        else:
            label = np.zeros(len(G.nodes))
        for index,node in enumerate(G.nodes):
            label[index] = G.nodes[node]['labels'][0]
        return label

    def get_label_edges(self, G, dic = False):
        if dic:
            label = {}
        else:
            label = np.zeros((len(G.nodes),len(G.nodes)))
        for e in G.edges:
            label[(e[0],e[1])] = G.edges[e]['labels'][0]
        return label


    def join(self,A,B):
        C = A.reshape(-1,1) - B.reshape(1,-1)
        return (C==0).astype(int)

        
    def adj_mask(self,A,B):
        n = int(A.shape[0]**0.5)
        m = int(B.shape[1]**0.5)
        C= (A*B).todense()# to have (n*n)x(m*m) where A[i,j] = 1 if e_i exist and e_j exist
        
        C = np.expand_dims(C,[1,3]).reshape(n,n,m,m)
        C = np.transpose(C,(0,2,1,3)).reshape(n*m,n*m)
        return C # return (n*m)x(n*m) where A[i,j] = 1 if (ei[0] -> ej[0] if an edge) as well as (ei[1] -> ej[1] if an edge

    def label_mask(self,A,B):
        C = self.join(A,B) #get C such that C[i,j] = 1 iff ni == nj
        return C.reshape(-1,1) * C.reshape(1,-1) # return (n*m)x(n*m) where A[i,j] = 1 if (ei[0] == ej[0]) and (ei[1] == ej[1])

    def edge_mask(self,A,B):
        C = self.join(A,B)
        n = A.shape[0]
        m = B.shape[0]
        C = np.expand_dims(C,[1,3]).reshape(n,n,m,m)
        C = np.transpose(C,(0,2,1,3)).reshape(n*m,n*m)
        return C

    

    def test(self,x):
        x = 0
    
    def pairwise_operation(self,G1,G2):
        ed_mas = self.edge_mask(G1[3],G2[3])
        lab_mask = self.label_mask(G1[2],G2[2])
        adj_mas = self.adj_mask(G1[0],G2[1])
        #tensor_adj = sparse.csc_array(adj_mas)#*lab_mask)
        tensor_adj = adj_mas*lab_mask*ed_mas
        if self.lamb is not None:
            tensor_adj = sparse.csc_array(tensor_adj)
            I = identity(tensor_adj.shape[0]) -self.lamb*tensor_adj
            x = spsolve(I,np.ones(tensor_adj.shape[0]))
        else:
            x = tensor_adj**self.n
        
        return x.sum()
    
    def kernel_computation(self, indices):
        i = indices[0]
        j = indices[1]
        G1 = self.X[i]
        G2 = self.X[j]
        if self.dic:
            self.K[i,j]= self.walk.pairwise_operation(G1,G2)
            self.K[j,i] = self.K[i,j]
        else:
            self.K[i,j] = self.pairwise_operation(G1,G2)
            self.K[j,i] = self.K[i,j]
        self.pbar.update(1)


    def kernel_computation(self,indices,idx, resume = -1, kernel = 'kernel',Kr = []):
        print(f"process {idx} start")
        K = Kr
        step = -1
        for ind in tqdm(indices):
            step += 1

            if step <= resume:
                continue
            i = ind[0]
            j = ind[1]
            G1 = self.X[i]
            G2 = self.X[j]
            if self.dic:
                K.append(self.walk.pairwise_operation(G1,G2))
                #K[j,i] = self.K[i,j]
            else:
                K.append(self.pairwise_operation(G1,G2))
                #self.K[j,i] = self.K[i,j]
            if step%15000 == 0 and step != 0:
                with open(f"{kernel}_{idx}_chpt.pkl", "wb")  as f:
                    pickle.dump({
                        "step":step,
                        "K":K
                    },f)

        return K
    
            
    def fit_transform(self,X,n=10, lamb = None, i = 0, m = 1,resume =-1, max_step = -1, kernel= 'kernel',Kr = []):
        self.K = np.zeros((len(X),len(X)))
        self.lamb = lamb
        self.n = n
        print("getting good format ...")
        self.X  = self.good_format(X,self.dic)
        if self.dic:
            self.walk = RandomWalkLabeled(lamda=lamb,p=n)
            self.X = self.walk.parse_input(X)
        indices = [(a,b) for a  in range(len(X)) for b in range(len(X)) if b >= a]
        if max_step == -1:
            max_step = len(indices)
        n_chunk = int((len(indices))/m) + 1
        chunk_indices = []
        for j in range(m):
            chunk_indices.append(indices[j*n_chunk:j*n_chunk+ n_chunk])
        
        
        chunk_res = self.kernel_computation(chunk_indices[i][:max_step],i,resume,kernel,Kr)

        with open(f"{kernel}_{i}_{m}.pkl", "wb")  as f:
            pickle.dump({
                "indices":chunk_indices[i],
                "K":chunk_res,
                "step":len(chunk_indices[i][:max_step])-1
            },f)
        
        # processes = []
        # manager = multiprocessing.Manager()
        # return_dict = manager.dict()
        # start = time.time()
        # for i,chunk in enumerate(chunk_indices):
        #     process = multiprocessing.Process(target=self.kernel_computation, args=(chunk,i,return_dict))
        #     process.daemon = True
        #     process.start()
        #     processes.append(process)
        
        # for process in processes:
        #     process.join()
        # duree = time.time() -start
        # print(duree)
        
        # for idx, chunk in enumerate (chunk_indices):
        #     res = return_dict[idx]
        #     for idx2, e in enumerate(chunk):
        #         self.K[e] = res[idx2]
        #         self.K[e[1],e[0]] = res[idx2]
        #breakpoint()
    def load_kernel(self,n,kernel = 'kernel',m = 1):
        K = np.zeros((n,n))
        for i in range(m):
            with open(f"{kernel}_{i}_{m}.pkl","rb") as f:
                temp = pickle.load(f)
                Kv = temp['K']
                indices = temp['indices']
            if len(Kv) == len(indices):
                for j,idx in enumerate(indices):
                    K[idx] = Kv[j]
                    K[idx[1],idx[0]] = Kv[j]
            else:
                with open(f"{kernel}_{i}_chpt.pkl", "rb") as f:
                    temp = pickle.load(f)
                    Kc = temp['K']
                    step = temp['step']
                assert len(Kv) + len(Kc) == len(indices)
                for j,idx in enumerate(indices):
                    if j<= step:
                        K[idx] = Kc[j]
                        K[idx[1],idx[0]] = Kc[j]
                    else:
                        K[idx] = Kv[j-step -1]
                        K[idx[1],idx[0]] = Kv[j-step -1]
        return K 
            
                
            
       



        