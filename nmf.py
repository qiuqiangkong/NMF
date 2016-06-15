'''
SUMMARY:  Non negative matrix (NMF)
          With Euclidean distance, KL divergence, beta divergence
AUTHOR:   Qiuqiang Kong
          q.kong@surrey.ac.uk
Liscence: BSD
Created:  2016.06.14
Modified: -
--------------------------------------
Params:   V         F*N matrix, F, N is number of features, samples
          n_basis   rank of decomposition
          n_iter    number of iteration
          norm_W    type of normalization on W, can be 0 (None), 1 (l1-norm), 2 (l2-norm)
          norm_H    type of normalization on H, can be 0 (None), 1 (l1-norm), 2 (l2-norm)
          W0        initial value of W, (default: random)
          H0        initial vlaue of H, (default: random)
          update_W  whether update W, (default: True)
          update_H  whether update H, (default: True)
return:   W         learned dictionary, size: F*K, where F, K is number of features, rank of decomposition
          H         learned representation, size: K*N, where K, N is rank of decomposition, number of samples
--------------------------------------
'''
import numpy as np

# normalize matrix by row, norm can be 1 or 2
def normalize( X, norm_X ):
    if norm_X==1:
        X /= np.sum( np.abs(X), axis=0 )
    if norm_X==2:
        X /= np.sqrt( np.sum( X*X, axis=0 ) )
    return X

'''
Euclidean distance. You can also use nmf_beta (beta=2), which is same.
[1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." 
Advances in neural information processing systems. 2001.
'''
def nmf_euc( V, n_basis, n_iter=10, norm_W=0, norm_H=0, W0=None, H0=None, update_W=True, update_H=True ):
    (F, N) = V.shape
    K = n_basis
    
    if W0 is None: W = 0.1 * np.random.rand( F, K )
    else: W = W0
    if H0 is None: H = 0.1 * np.random.rand( K, N )
    else: H = H0
    
    for n in xrange( n_iter ):
        if update_W is True: W = W * np.dot( V, H.T ) / np.dot( np.dot( W, H ), H.T )
        if update_H is True: H = H * np.dot( W.T, V ) / np.dot( np.dot( W.T, W ), H )
        if norm_W!=0: W = normalize( W, norm_W )
        if norm_H!=0: H = normalize( H, norm_H )
        
    return W, H

'''
KL divergence. You can also use nmf_beta (beta=1), which is same.
You can also use nmf_beta (beta=1), which is same.
[1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." 
Advances in neural information processing systems. 2001.
'''    
def nmf_kl( V, n_basis, n_iter=10, norm_W=0, norm_H=0, W0=None, H0=None, update_W=True, update_H=True ):
    (F, N) = V.shape
    K = n_basis
    
    if W0 is None: W = 0.1 * np.random.rand( F, K )
    else: W = W0
    if H0 is None: H = 0.1 * np.random.rand( K, N )
    else: H = H0
    
    for n in xrange( n_iter ):
        if update_W is True: W = W * ( np.dot( V / np.dot( W, H ), H.T ) / np.sum( H, axis=1 ) )
        if update_H is True: H = H * ( np.dot( W.T, V / np.dot( W, H ) ) / np.sum( W, axis=0 )[:,None] )
        if norm_W!=0: W = normalize( W, norm_W )
        if norm_H!=0: H = normalize( H, norm_H )
        
    return W, H
    
'''
Beta divergence. 
[1] Fevotte, Cedric, Nancy Bertin, and Jean-Louis Durrieu. "Nonnegative matrix factorization with the 
Itakura-Saito divergence: With application to music analysis." Neural computation 21.3 (2009): 793-830.
'''
def nmf_beta( V, n_basis, beta, n_iter=10, norm_W=0, norm_H=0, W0=None, H0=None, update_W=True, update_H=True ):
    (F, N) = V.shape
    K = n_basis
    
    if W0 is None: W = 0.1 * np.random.rand( F, K )
    else: W = W0
    if H0 is None: H = 0.1 * np.random.rand( K, N )
    else: H = H0
    
    for n in xrange( n_iter ):
        if update_W is True: W = W * np.dot( np.power( np.dot( W, H ), beta-2 ) * V, H.T ) / ( np.dot( np.power( np.dot( W, H ), beta-1 ), H.T ) )
        if update_H is True: H = H * ( np.dot( W.T, ( np.power( np.dot( W, H ), beta-2 ) * V ) ) ) / ( np.dot( W.T, np.power( np.dot( W, H ), beta-1 ) ) )
        if norm_W!=0: W = normalize( W, norm_W )
        if norm_H!=0: H = normalize( H, norm_H )
        
    return W, H