import sys
import numpy as np

def compute_class_precision_recall(L,K):
    _,L = np.unique(np.array(L),return_inverse=True)
    _,K = np.unique(np.array(K),return_inverse=True)
    if(len(L) != len(K)):
        sys.stderr.write("Labels and clusters are not of the same length.")
        sys.exit(1)
    num_elements = len(L)
    num_labels   = L.max() + 1
    num_clusters = K.max() + 1
    X_L = np.tile(L, (num_labels,1) ).T
    X_K = np.tile(K, (num_clusters,1) ).T
    L_j = np.equal( np.tile(np.arange(num_labels),(num_elements,1))   , X_L ).astype(float)
    K_j = np.equal( np.tile(np.arange(num_clusters),(num_elements,1)) , X_K ).astype(float)
    P_ij = np.dot(L_j.T,K_j)
    S_i  = P_ij.sum(axis=1)
    T_i  = P_ij.sum(axis=0)
    R_i  = ( P_ij * P_ij ).sum(axis=1) / ( S_i * S_i )
    P_i  = ( P_ij.T * P_ij.T ).sum(axis=1) / ( T_i * T_i )
    return [(P_i , R_i) , (S_i , T_i)]

def calc_b3(L , K , class_norm=False, beta=1.0):
    precision_recall , class_sizes = compute_class_precision_recall(L,K)
    if(class_norm == True):
        precision = precision_recall[0].sum() / class_sizes[1].size
        recall    = precision_recall[1].sum() / class_sizes[0].size
    else:
        precision = ( precision_recall[0] * class_sizes[1] ).sum() / class_sizes[1].sum()
        recall    = ( precision_recall[1] * class_sizes[0] ).sum() / class_sizes[0].sum()
    f_measure = (1 + beta**2) * (precision * recall) /( (beta**2) * precision + recall )
    return [f_measure,precision,recall]

# L = np.array([1,3,3,3,3,4,2,2,0,3,3])
# K = np.array([1,2,3,4,5,5,5,6,2,1,1])
# # Standard BCUBED
# [fmeasure, precision, recall] = calc_b3(L,K)
#
# print( calc_b3(L,K) )
