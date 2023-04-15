import pickle
import numpy as np
import networkx as nx
from kernel_model import sample_weight

from random_walk import RandonWalk
import multiprocessing as mp
from model import classifier
from dataviz import plot_PCA
from model import classifier
import argparse

parser = argparse.ArgumentParser('generate final submission')
parser.add_argument('--data_path', help='folder for extracted data')
parser.add_argument('--output', help="name of output file")



if __name__ =="__main__":
    mp.set_start_method('spawn')
    print("loading data...")
    args = parser.parse_args()
    with open(f"{args.data_path}/training_data.pkl","rb") as train_data, open(f"{args.data_path}/test_data.pkl","rb") \
        as test_data, open(f"{args.data_path}/training_labels.pkl","rb") as train_label:
        train = pickle.load(train_data)
        test = pickle.load(test_data)
        label = pickle.load(train_label)

    N_train = len(train)
    N_test = len(test)

    N_train_2 = int(0.8*N_train)
    N_val = N_train - N_train_2

    print("building kernel...")
    from kernel import kernel_edge,kernel_node,kernel_len,kernel_form,kernel_neighbors, kernel_adj
    max_label_node = 50
    max_label_edge = 4

    whole = train + test
    walk = RandonWalk(dic=False)
    #walk.fit_transform(whole,n=15,lamb=None,kernel="kernel_test")
    K1W = kernel_edge(whole,max_label_node,max_label_edge,var=25)
    K2W = kernel_node(whole,max_label_node, var = 25)
    K3W = kernel_len(whole, var=150)
    K5W = kernel_neighbors(whole,max_label_node,200,30)
    #K4W = walk.load_kernel(n = len(whole),kernel='kernel_test')


    K1 = K1W[:N_train,:][:,:N_train]
    K2 = K2W[:N_train,:][:,:N_train]
    K3 = K3W[:N_train,:][:,:N_train]
    #K4 = K4W[:N_train,:][:,:N_train]
    K5 = K5W[:N_train,:][:,:N_train]


    K1_test = K1W[-N_test:,:][:,:N_train]
    K2_test = K2W[-N_test:,:][:,:N_train]
    K3_test = K3W[-N_test:,:][:,:N_train]
    #K4_test = K4W[-N_test:,:][:,:N_train]
    K5_test = K5W[-N_test:,:][:,:N_train]

    

   
    


    #lr2 = classifier('ridge',class_weight = 'balanced',alpha=2,w=[1,1], tr = 0.01, tr2= 0.01,tr3=[0.5,0.5])
    lr1 = classifier('ridge2',class_weight = 'balanced',C=0.005,tr = 0.01, tr2= 0.01,w=[1,1])
    
    K = K5*(0.5*K1*K2 + 0.5*K2*K3) #+ 0.08*K4
    print("cross validation score for ridge")
    #score = lr1.cross_val(K ,label, cv=4, ratio= 0.8, balanced=False)

    lr1.train(K,label)
    lr1.submit(K, args.output)