import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow import keras
import numpy as np

def get_k(evals, delta=0.99):
    """Determine minimal number k (+1) of largest evals neccessary for spanning a 
    variance eigenspace of delta.
    k := argmax(sum(k_largest_evals)/N) <= delta
    
    {"list"(tensor) of evals -> [sort] -> [cumsum] cumulative evals 
    -> [divide] by 'N' (covmat-trace/layer variance) -> [var_ratio_evs<=delta] cond (1/0 booleans) 
    -> [rsum] (scalar) number cum_evs<=d -> k   [+track corresponding ixs]}
    '<=' is checked for all list-ixs (avoids looping in the Graph) at once, so k is found 
    by rsum over (boolean True/1) elems of the bool-list cond. (less hacky than it looks)
        
    Returns: k and the indices of the k largest eigenvalues.         
    """
    #assert evals.dtype == tf.float64, "EigVals not f64 in llk"
    
    #Sort in DESC. Order 
    s_evs = tf.sort(evals, axis=0, direction='DESCENDING', name="sort_ev") 
    s_ixs = tf.argsort(evals, axis=0, direction='DESCENDING', stable=True, name="sort_ev_ix")  
    
    #Cumulative sums of largest EVs
    cum_evs = tf.cumsum(s_evs, axis=0, name="cumsum_evals") 
    
    #Ratios Cum.sums to total sum of EVs (trace(CovMat) ~ variance of CovMat)
    dim_ratios = tf.realdiv(cum_evs, cum_evs[-1], name="div_by_trace")
    
    #Check what ratios <= delta
    cond = tf.math.less_equal(dim_ratios, delta) #TODO change first F ix
    
    k_s_ixs = tf.boolean_mask(s_ixs, cond)
    
    k = tf.math.maximum(tf.dtypes.cast(len(tf.where(cond)), dtype=tf.float64) , tf.constant([1.], dtype=tf.float64))
    return k, k_s_ixs

def cov_mat(o_s, sum_sqrs, runn_sum):
    
    #SAMPLE MEAN 
    sample_mean = tf.realdiv(runn_sum, o_s, name='sample_mean') #alt tf.math.scalar_mul()
    
    #DOT PRODUCT OF SAMPLE MEAN
    sample_mean_dot = tf.tensordot(sample_mean, sample_mean, axes=0, name='sample_mean_dot') #alt tf.einsum()

    #COVARIANCE MATRIX
    cov_mat = tf.math.subtract(tf.realdiv(sum_sqrs, o_s), sample_mean_dot, name='cov_matrix') 
    
    return cov_mat
    
def saturation(cov_mat, features):
    
    #EIGEN DECOMPOSITION 
    eig_vals_raw, eig_vecs_raw = tf.linalg.eigh(cov_mat, name='eig_decomp') 
    eig_vals = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vals_raw)), dtype=tf.float64, name="eig_vals_real")
    eig_vecs = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vecs_raw)), dtype=tf.float64, name="eig_vecs_real")#why convert_to_tensor again? eig shld rtrn that 
    
    #K
    k, ixs = get_k(eig_vals, delta=0.99)

    #PROJECTION MATRIX  TODO alt. tf.slice + tune indices (also for batches)
    k_eig_vecs = tf.gather(params=eig_vecs, indices=ixs , axis=0, batch_dims=1, name="k_eig_vecs") 
    proj_mat = tf.matmul(k_eig_vecs, k_eig_vecs, transpose_a=True, name="proj_mat")

    saturation = tf.truediv(k, features, name="saturation")
    
    return saturation 