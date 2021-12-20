import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow import keras
import numpy as np

#TODO check where s_s come before r_s and fix it 
def get_cov_mat(o_s, runn_sum, sum_sqrs):
    """
        NAIVE ALGORITHM
        
        Gets Sample Covariance-Matrix Estimate from given values for the three naive Algorithm states.
        
        Parameters: 
        
            o_s (scalar tf.tensor): count of samples observed so far
            runn_sum (tf.tensor): sum of layer activations so far
            sum_sqrs (tf.tensor): sum of squares of layer activations so far
            
        Returns:
        
            cov_mat (tf.tensor): Estimated Covariance-matrix of the layer    
    """
    
    #SAMPLE MEAN 
    sample_mean = tf.realdiv(runn_sum, o_s, name='sample_mean') #alt tf.math.scalar_mul()
    
    #DOT PRODUCT OF SAMPLE MEAN
    sample_mean_dot = tf.tensordot(sample_mean, sample_mean, axes=0, name='sample_mean_dot') #alt tf.einsum()

    #COVARIANCE MATRIX
    cov_mat = tf.math.subtract(tf.realdiv(sum_sqrs, o_s), sample_mean_dot, name='cov_matrix') 
    
    return cov_mat

def get_eig(cov_mat):
    """
        Computes the Eigenvalue/Eigenvector pairs for a given Covariance-matrix.
        
        Parameters:
            cov_mat (2D tf.tensor): nxn Covariance-matrix 
        
        Returns:
            eig_vals (tf.tensor): "listed" n Eigenvalues in non-decreasing order (-> necessary for estimating saturation)
            eig_vecs (tf.tensor): "listed" n Eigenvectors in corresponding(?) order (-> necessary for projecting layer)
            
    """
    
    #EIGEN DECOMPOSITION 
    eig_vals_raw, eig_vecs_raw = tf.linalg.eigh(cov_mat, name='eig_decomp') 
    eig_vals = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vals_raw)), dtype=tf.float64, name="eig_vals_real")
    eig_vecs = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vecs_raw)), dtype=tf.float64, name="eig_vecs_real")#why convert_to_tensor again? eig shld rtrn that 
    
    return eig_vals, eig_vecs    

def get_k(evals, delta=0.99):
    """
        Finds minimal number k, so that: 
            Ratio between (Sum of k+1 largest Eigenvalues) and (total layer variance) exceeds threshold delta. 
            ( = Fraction of accounted-for over total layer variance.)
            (<-> k Eigenvectors span a variance eigenspace for the given delta)
            
        Condition as Formula:
            k := argmax(sum(k_largest_evals)/N) <= delta #TODO /N ? Flüchtigkeitsfehler?
        
        Process outlined:    
            evals -> [sort: largest first] -> [cumsum] => cumulative sum of evals 
            -> [divide by total eval-sum] => ratios of largest evals to covmat-trace/layer variance 
            -> [cond ratio<=delta] => bool list -> [len(where())] => number of ratios <= delta  
            -> [max(true-ratios,1)] => number k eigvals (k >= 1)  [+track ixs]
                
            Computing k by counting ratios satisfying  "<=" from all ratios avoids branches/loops, 
            a possible source of problems in TensorFlows Graph Mode.
            
        Returns: 
            k, idxs of eigvals #TODO just count k eigvecs          
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
    """
        NAIVE ALGORITHM
        
        Gets Saturation from values for naive Algorithm states
        by performing/including all single steps in order. 
        Calls functions:  get_cov_mat(), get_eig(), get_k()
        
        Parameters: 
        
            features (scalar tf.tensor): number of layer features
            o_s (scalar tf.tensor): number of counted samples so far
            r_s (tf.tensor): running sum of layer activations so far
            s_s (tf.tensor): sum of squared activations so far
            delta (float): value for threshold on the fraction of "explained" variance

        Returns: 
            
            saturation (scalar tf.tensor): saturations value resulting from inputs        
    """
    
    cov_mat = get_cov_mat(o_s, r_s, s_s)
    evals, evecs = get_eig(cov_mat)
    k, _ixs = get_k(evals, delta) 
    
    return tf.truediv(k, features, name="saturation") 
    
def saturation(cov_mat, features, delta=0.99):
    """
        Gets Saturation from given Covariance-matrix, number of features (layer width), and delta value.
        
        Parameters:
            cov_mat (tf.tensor): any Covariance-matrix 
            features (int???): number of units in layer (width) #TODO int?
            delta (float): chosen threshold for "lossless" retract
        
        Returns:
            saturation (tf.tensor): computed saturation
            
            
    """
    #EIGEN DECOMPOSITION 
    eig_vals_raw, eig_vecs_raw = tf.linalg.eigh(cov_mat, name='eig_decomp') 
    eig_vals = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vals_raw)), dtype=tf.float64, name="eig_vals_real")
    eig_vecs = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vecs_raw)), dtype=tf.float64, name="eig_vecs_real")#why convert_to_tensor again? eig shld rtrn that 
    
    #K
    k, ixs = get_k(eig_vals, delta=delta)

    #Not used currently:
    #PROJECTION MATRIX  #TODO alt. tf.slice + tune indices (also for batches)
    k_eig_vecs = tf.gather(params=eig_vecs, indices=ixs , axis=0, batch_dims=1, name="k_eig_vecs") 
    proj_mat = tf.matmul(k_eig_vecs, k_eig_vecs, transpose_a=True, name="proj_mat")

    saturation = tf.truediv(k, features, name="saturation")
    
    return saturation 

def get_update_values(self, input_tensor):
    """
        NAIVE ALGORITHM 
        
        Compute and return update values for Naive Algorithm states 
        from single activation tensor without affecting states. 
        #TODO -> probably the self argument can be deleted unless it was a hack. was there sth. hacky here?
        
        -> Mostly for Metric approach. 
        #TODO Probably Obsolete right? Delete/Archive?  <-- <-- <--
        
        
        Parameters:
        
            self : Unnecessary??
            input_tensor (tf.tensor): the layer batch-activation to update states with
            
        Returns:
        
            batch_size (tuple): number of newly observed samples to update o_s with #TODO should be tensor instead of tuple right?     
            sum_over_batch (tf.tensor): Sum of activations over the batch to update running_sum with
            sqr_inp (tf.tensor): Square(~~hmm) of the batch-activation tensor to update sum_sqrs with
        
    """
    
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

def two_pass_cov(activations):
    """
        TWO-PASS ALGORITHM
        
        Get Covariance-matrix Estimate using 2-Pass Algorithm with *previously logged* 
        list of batch-activations of the measurement window (i.e. one epoch) as input.
        
        Formula:
            Sum( (x - x_mean)(y - y_mean) ) / N
        
        
        
        TODO test both final steps divided and in one
        TODO test -> dims stimmen
        TODO timeit
        TODO compare to naive algorithm covmat
        #should pass activations as one big numpy array sth. to_tensor bug slow
        
        Parameters:
            activations (tf.tensor): "listed" batch-activations of one epoch/measurement-window
        
        #TODO check how this function was meant to work and what the two return values are lol
        Returns: 
            cov_mat_from_batches (): ?
            cov_mat_from_epoch (): ?
            
    """
    #activations = tf.convert_to_tensor(np.asanyarray(activations))
    #print(f"")
    batch_shape = activations[0].shape
    epoch_activations = tf.concat(activations, axis=0) #one big batch-dim
    print(f"epoch_activations: {epoch_activations.get_shape()}")
    N = tf.dtypes.cast(epoch_activations.shape[0], dtype=tf.float64)
    print(f"N: {N}")
    
    total_sum = tf.reduce_sum(epoch_activations, axis=0)
    total_mean = total_sum / N
    print(f"total_mean: {total_mean.get_shape()}")
    
    #probably more memory efficient to still do sum of squares
    #divided in batches? try it^^
    broadcast_mean = tf.broadcast_to(total_mean, batch_shape)
    epoch_batches = tf.stack(activations, axis=0)
    print(f"epoch_batches: {epoch_batches.get_shape()}")
    norm_batches = epoch_batches -  broadcast_mean #still broadcast again, just do it implicitly
    square_batches = tf.map_fn(fn=lambda m: tf.linalg.matmul(m,m,transpose_a=True) / N, elems=norm_batches)
    print(f"square_batches: {square_batches.get_shape()}")
    mean_square = tf.math.reduce_sum(square_batches, axis=0)
    #cov_mat = tf.linalg.matmul(norm_batches, norm_batches, transpose_a=True) / N
    cov_mat_from_batches = mean_square
    
    #return cov_mat_from_batches 
    
    #Without batches
    norm_epoch = epoch_activations - total_mean #should broadcast over sample-dim
    cov_mat_from_epoch = tf.linalg.matmul(norm_epoch, norm_epoch, transpose_a=True) / N
    
    #return cov_mat_from_epoch
    
    return cov_mat_from_batches, cov_mat_from_epoch