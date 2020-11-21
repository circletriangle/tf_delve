import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow import keras
import numpy as np

#TODO check where s_s come before r_s and fix it 
def get_cov_mat(o_s, runn_sum, sum_sqrs):
    
    #SAMPLE MEAN 
    sample_mean = tf.realdiv(runn_sum, o_s, name='sample_mean') #alt tf.math.scalar_mul()
    
    #DOT PRODUCT OF SAMPLE MEAN
    sample_mean_dot = tf.tensordot(sample_mean, sample_mean, axes=0, name='sample_mean_dot') #alt tf.einsum()

    #COVARIANCE MATRIX
    cov_mat = tf.math.subtract(tf.realdiv(sum_sqrs, o_s), sample_mean_dot, name='cov_matrix') 
    
    return cov_mat

def get_eig(cov_mat):
    #EIGEN DECOMPOSITION 
    eig_vals_raw, eig_vecs_raw = tf.linalg.eigh(cov_mat, name='eig_decomp') 
    eig_vals = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vals_raw)), dtype=tf.float64, name="eig_vals_real")
    eig_vecs = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vecs_raw)), dtype=tf.float64, name="eig_vecs_real")#why convert_to_tensor again? eig shld rtrn that 
    
    return eig_vals, eig_vecs    

def get_k(evals, delta=0.99):
    """Find min. number k, so k+1 largest evals account for delta of layer variance. 
    (spanning a variance eigenspace of delta)
    
    k := argmax(sum(k_largest_evals)/N) <= delta
    
    evals -> [sort: largest first] -> [cumsum] => cumulative sum of evals 
    -> [divide by total eval-sum] => ratios of largest evals to covmat-trace/layer variance 
    -> [cond ratio<=delta] => bool list -> [len(where())] => number of ratios <= delta  
    -> [max(true-ratios,1)] => number k eigvals (k >= 1)  [+track ixs]
    
    Computing k by counting ratios satisfying  "<=" from all ratios avoids branches/loops, 
    a possible source of problems in TensorFlows Graph Mode.
      
    Returns: k, idxs of eigvals TODO just count k eigvecs          
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
    
    k_pre = tf.dtypes.cast(tf.size(tf.where(cond)), dtype=tf.float64)
    k = tf.math.maximum(k_pre, tf.constant([1.], dtype=tf.float64))
    
    #can't use len() in graph mode TODO assert both k match in eager
    #k = tf.math.maximum(tf.dtypes.cast(len(tf.where(cond)), dtype=tf.float64) , tf.constant([1.], dtype=tf.float64))
    
    return k, k_s_ixs
 
def get_sat(features, o_s, r_s, s_s, delta=0.99):
    """Get Saturation from states (including all steps)."""
    
    cov_mat = get_cov_mat(o_s, r_s, s_s)
    evals, evecs = get_eig(cov_mat)
    k, _ixs = get_k(evals, delta) 
    
    return tf.truediv(k, features, name="saturation") 
    
def saturation(cov_mat, features, delta=0.99):
    """Get Saturation from finished Cov-Mat."""
    #EIGEN DECOMPOSITION 
    eig_vals_raw, eig_vecs_raw = tf.linalg.eigh(cov_mat, name='eig_decomp') 
    eig_vals = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vals_raw)), dtype=tf.float64, name="eig_vals_real")
    eig_vecs = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vecs_raw)), dtype=tf.float64, name="eig_vecs_real")#why convert_to_tensor again? eig shld rtrn that 
    
    #K
    k, ixs = get_k(eig_vals, delta=delta)

    #PROJECTION MATRIX  TODO alt. tf.slice + tune indices (also for batches)
    k_eig_vecs = tf.gather(params=eig_vecs, indices=ixs , axis=0, batch_dims=1, name="k_eig_vecs") 
    proj_mat = tf.matmul(k_eig_vecs, k_eig_vecs, transpose_a=True, name="proj_mat")

    saturation = tf.truediv(k, features, name="saturation")
    
    return saturation 

def get_update_values(self, input_tensor):
    """Compute update values without changing the states. Mostly for Metric approach"""
    
    #OBSERVED SAMPLES
    shape = input_tensor.get_shape()#get_shape might fail
    batch_size = (shape[0]) 
    if not batch_size:
        batch_size=(0.)
    
    #RUNNING SUM
    input_tensor = tf.dtypes.cast(input_tensor, dtype=tf.float64)
    sum_over_batch = tf.reduce_sum(input_tensor, axis=0)  
    
    #SUM OF SQUARES
    sqr_inp = tf.linalg.matmul(input_tensor, input_tensor, transpose_a=True)   
    
    return batch_size, sum_over_batch, sqr_inp

