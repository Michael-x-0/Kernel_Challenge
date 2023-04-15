import numpy as np
import networkx as nx

def kernel_neighbors(X,max_label_node,max_neighbors, var = 1):
    B = np.zeros((len(X),max_label_node,max_neighbors))
    for k,G in enumerate(X): 
        for p,x in enumerate(G.nodes):
            nei = list(G.neighbors(p))
            i = G.nodes[x]['labels'][0]
            B[k,i,nei] += 1

    #B = np.sum(B**2, axis=1).reshape(-1,1) -2*B@B.T +np.sum(B**2, axis=1).reshape(1,-1)
    #B = np.exp(-B/var)
    B = B.reshape(len(X),max_label_node*max_neighbors)
    B = np.sum(B**2, axis=1).reshape(-1,1) -2*B@B.T +np.sum(B**2, axis=1).reshape(1,-1)
    B = np.exp(-B/var)
    return B

def kernel_adj(X,max_node, var = 1):
    B = np.zeros((len(X),max_node))
    for k,G in enumerate(X): 
        A = nx.adjacency_matrix(G).todense()
        l = min(A.shape[0],max_node)
        A =  A[:l,:l]
        value,_ = np.linalg.eig(A)
        value = np.sort(np.real(value))[::-1]
        B[k,:l] =value
    #B = B.reshape(len(X),-1)
    B = np.sum(B**2, axis=1).reshape(-1,1) -2*B@B.T +np.sum(B**2, axis=1).reshape(1,-1)
    B = np.exp(-B/var)
    return B


    #B = np.sum(B**2, axis=1).reshape(-1,1) -2*B@B.T +np.sum(B**2, axis=1).reshape(1,-1)
    #B = np.exp(-B/var)
    B = B.reshape(len(X),max_label_node*max_neighbors)
    B = np.sum(B**2, axis=1).reshape(-1,1) -2*B@B.T +np.sum(B**2, axis=1).reshape(1,-1)
    B = np.exp(-B/var)
    return B@B.T


def kernel_edge(X, max_label_node, max_label_edge,var = 1):
    B = np.zeros((len(X),(max_label_edge+1)*max_label_node*max_label_node))
    for p,G in enumerate(X):
        A = np.zeros((max_label_node,max_label_node,max_label_edge + 1))
        for x in G.edges:
            i = G.nodes[x[0]]['labels'][0]
            j = G.nodes[x[1]]['labels'][0]
            temp = max(i,j)
            i = min(i,j)
            j = temp
            k = G.edges[x]['labels'][0] + 1
            A[i,j,k] += 1
            A[i,j,0] = 1
            A[i,i,0] = 1
        B[p,:] = A.flatten()
    B = np.sum(B**2, axis=1).reshape(-1,1) -2*B@B.T +np.sum(B**2, axis=1).reshape(1,-1)
    B = np.exp(-B/var)
    return B

def kernel_form(X, max_label_node, max_label_edge, var = 1):
    B = np.zeros((len(X),1))
    for p,G in enumerate(X):
        A = np.zeros((max_label_node,max_label_node,max_label_edge))
        for x in G.edges:
            i = G.nodes[x[0]]['labels'][0]
            j = G.nodes[x[1]]['labels'][0]
            temp = max(i,j)
            i = min(i,j)
            j = temp
            k = G.edges[x]['labels'][0]
            A[i,j,k] += 1
        A = np.sum(A, axis=2)
        L = np.diag(np.sum(A, axis = 1))-A
        e = np.linalg.eig(L)[0]
        B[p,0] = np.sum(e==0)
    B = np.sum(B**2, axis=1).reshape(-1,1) -2*B@B.T +np.sum(B**2, axis=1).reshape(1,-1)
    B = np.exp(-B/var)

        
    return B

def kernel_node(X, max_label_node, var = 1):
    A = np.zeros((len(X), max_label_node))
    for i,x in enumerate(X):
        for node in x.nodes:
            A[i,x.nodes[node]['labels'][0]] += 1
    A= np.sum(A**2, axis=1).reshape(-1,1) -2*A@A.T +np.sum(A**2, axis=1).reshape(1,-1)
    A = np.exp(-A/var)
    return A

def kernel_len(X,var = 4):
    A = np.zeros((len(X),4))
    for i,G in enumerate(X):
        A[i,0] = len(G.nodes)
        A[i,1] = len(G.edges)
        A[i,2] = len(G.edges)/len(G.nodes)
        #A[i,3] = nx.number_connected_components(G)
    #(A - A.mean(axis = 0))/(A.std(axis=0))
    K = np.sum(A**2, axis=1).reshape(-1,1) -2*A@A.T +np.sum(A**2, axis=1).reshape(1,-1)
    K = np.exp(-K/var)
    return K