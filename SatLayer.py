import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow import keras
import numpy as np

"""
Class for computing the Saturation of a keras Layer implemented as a 
custom keras Layer to be attached to the actual layer.
Functionally similar to stateful keras metrics relevant states are 
updated in update(), reset in reset() and queried with result().
The core computation of Saturation given the states is done in sat(). 

based on SatMod_modstates_current.py 15.10.2020 2:21
to be used by Clone_Model extended Layers as sublayer

"""



class sat_layer(keras.layers.Layer):
    """
    TODO maybe convert to keras metric now that injection is done?
    Intended Sublayer of Clone_Model.mydense Layers to compute the Saturation of the Layer. 
    Functions update/reset/result like keras metric (but not treated as a metric object).
    Tracks three states with update() of "Parent-layer"-activation in forward-pass, and reset() at epoch-end: 
    -Number of observed samples 
    -Running Sum (of activations) (NOT Running Mean, that's only computed when needed in sat())
    -Sum of Squares 
    """

    def lossless_k_loopless(self, evals):
        """Determine minimal number k (+1) of largest evals neccessary for spanning a 
        variance eigenspace of delta.
        k := argmax(sum(k_largest_evals)/N) <= delta
        TODO N := trace of Cov_mat?! (num_features (layer width)? probably not o_s?) 
        
        {"list"(tensor) of evals -> [sort] -> [cumsum] cumulative evals 
        -> [divide] by 'N' (covmat-trace/layer variance) -> [var_ratio_evs<=delta] cond (1/0 booleans) 
        -> [rsum] (scalar) number cum_evs<=d -> k   [+track corresponding ixs]}
        '<=' is checked for all list-ixs (avoids looping in the Graph) at once, so k is found 
        by rsum over (boolean True/1) elems of the bool-list cond. (less hacky than it looks)
           
        Returns: k and the indexes of the k largest eigenvalues.         
        """
        #assert evals.dtype == tf.float64, "EigVals not f64 in llk"
        
        #Sort in DESC. Order 
        s_evs = tf.sort(evals, axis=0, direction='DESCENDING', name="sort_ev") 
        s_ixs = tf.argsort(evals, axis=0, direction='DESCENDING', stable=True, name="sort_ev_ix")  
        #print("evals: {}\nevals_sorted_desc: {}".format(evals, s_evs)) 
        
        #Get list of cumulative sum of largest EVs
        cum_evs = tf.cumsum(s_evs, axis=0, name="cumsum_evals") 
        #print("Cumulating EVs: {}".format(cum_evs))
        
        #Divide Cum. sums by total sum of EVs (trace(CovMat) ~ variance of CovMat)
        dim_ratios = tf.realdiv(cum_evs, cum_evs[-1], name="div_by_trace")
        print("Dim ratios: {}".format(dim_ratios))
        
        cond = tf.math.less_equal(dim_ratios, self.delta) #TODO change first F ix
        #print("cond: {}".format(cond))
        
        k_s_ixs = tf.boolean_mask(s_ixs, cond)
        
        #Default to k=1 if first EVal explains delta of variance alone 
        #TODO change condition so graph mode works without tf.function
        if not tf.cond(cond[0], lambda: True, lambda: False):
            cond = tf.constant([True], dtype=tf.bool)
            #print("cond with default catch: {}".format(cond))   
            
        
        #k = tf.reduce_sum(tf.where(cond), axis=-1, name="k")
        k = len(tf.where(cond)) #TODO use min(1,tf.where()) instead of checking above
        
        #print("tf.where(): {}".format(tf.where(cond)))
        #print("K: in ll_fun {}".format(k))
        
        return k, k_s_ixs

    def lossless_k_loopless_comp(self, evals):
        """Determine minimal number k (+1) of largest evals neccessary for spanning a 
        variance eigenspace of delta.
        k := argmax(sum(k_largest_evals)/N) <= delta
           
        Returns: k and the indexes of the k largest eigenvalues.         
        """
        pr = 5
        
        #Invert order -> ascending real parts negative, should be desc. then (maybe difference abs/smaller than)
        s_evs = evals#tf.reverse(evals, axis=[-1], name="invert_comp_evs")
        #assert evals[2]==s_evs[-3], "inverting order problem"
        print("Evals head: {}\nEvals tail: {}".format(s_evs[:pr], s_evs[-pr:]))
        
        #Cumulative sum of largest EVs
        cum_evs = tf.cumsum(s_evs, axis=0, name="cumsum_evals") 
        print("Cumulating EVs head: {}\nCum EVs tail: {}".format(cum_evs[:pr], cum_evs[-pr:]))
        
        #Divide Cum. sums by total sum of EVs (trace(CovMat) ~ variance of CovMat)
        #dim_ratios = tf.realdiv(cum_evs, cum_evs[-1], name="div_by_trace")
        #print("Dim ratios: {}".format(dim_ratios)) # -> ratios w real parts slightly over 1, im still small ~e-20
        dim_ratios = tf.realdiv(tf.math.real(cum_evs), tf.math.real(cum_evs[-1]), name="div_by_total_variance_real")
        print("Dim ratios head: {}\nDim ratios: tail: {}".format(dim_ratios[:pr], dim_ratios[-pr:])) # -> ratios of real parts STILL slightly over 1
        #ratios over 1 -> either bc of im or sorting problem (Sorting problem)
        #
        
        
        cond = tf.math.less_equal(tf.math.real(dim_ratios), self.delta) #TODO change first F ix
        #print("cond: {}".format(cond))
        
        #k_s_ixs = tf.boolean_mask(s_ixs, cond)
        
        #Default to k=1 if first EVal explains delta of variance alone 
        #TODO change condition so graph mode works without tf.function
        if not tf.cond(cond[0], lambda: True, lambda: False):
            cond = tf.constant([True], dtype=tf.bool)
            #print("cond with default catch: {}".format(cond))   
            
        
        #k = tf.reduce_sum(tf.where(cond), axis=-1, name="k")
        k = len(tf.where(cond)) #TODO use min(1,tf.where()) instead of checking above
        
        #print("tf.where(): {}".format(tf.where(cond)))
        #print("K: in ll_fun {}".format(k))
        
        return k, k_s_ixs


    
    #@tf.function needed for bool usage in ll_k() but removes prints 
    def sat(self, o_s, sum_sqrs, runn_sum):
        """Compute Saturation given values (current or other) for the tracked states. Doesn't change states.
        
        Parameters: 
        o_s (scalar-shape tensor), sum_sqrs (shape(FxF) tensor), runn_sum (shape(F) tensor) : Given states
        
        Returns:
        scalar-shape tensor: saturation 
        """    
        
        #SAMPLE MEAN (from running sum of features and running sample count) (Not computed every batch)
        sample_mean = tf.realdiv(runn_sum, o_s, name='sample_mean') #alt tf.math.scalar_mul()
        #print(sample_mean)
        
        #DOT PRODUCT OF SAMPLE MEAN
        sample_mean_dot = tf.tensordot(sample_mean, sample_mean, axes=0, name='sample_mean_dot') #alt tf.einsum()

        #COVARIANCE MATRIX
        cov_mat = tf.math.subtract(tf.realdiv(sum_sqrs, o_s), sample_mean_dot, name='cov_matrix') 
        #print("CovMat with shape {}:\n {}".format(tf.shape(cov_mat),cov_mat))
        
        #EIGEN DECOMPOSITION TODO invert order here?, TODO check eig() as possible error source -> eigh?
        #abs() removes signs -> use tf.math.real() ignoring Im or use comp EVs 
        eig_vals_raw, eig_vecs_raw = tf.linalg.eigh(cov_mat, name='eig_decomp') 
        eig_vals = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vals_raw)), dtype=tf.float64, name="eig_vals_real")
        eig_vecs = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vecs_raw)), dtype=tf.float64, name="eig_vecs_real")#why convert_to_tensor again? eig shld rtrn that 
        #print("Eigenvalues shape: {}".format(tf.shape(eig_vals)))
        
       
        
        #K
        k, ixs = self.lossless_k_loopless(eig_vals) #abs.val compl ev #tf.constant(4, dtype=tf.int32)
        k_fl = tf.cast(k, dtype=tf.float64, name="k_fl")
        print("K tensor: {}".format(k_fl))

        #PROJECTION MATRIX  TODO alt. tf.slice + tune indices (also for batches)
        k_eig_vecs = tf.gather(params=eig_vecs, indices=ixs , axis=0, batch_dims=1, name="k_eig_vecs") 
        proj_mat = tf.matmul(k_eig_vecs, k_eig_vecs, transpose_a=True, name="proj_mat")

        saturation = tf.truediv(k_fl, self.layer_width_tensor, name="saturation")
        
       
        #ASSERTS dtypes
        for node in [sample_mean,
                     sample_mean_dot, cov_mat, eig_vals_raw, eig_vals,
                     eig_vecs, k_fl, k_eig_vecs, proj_mat, saturation]:
            assert node.dtype in [tf.float64, tf.complex128], "Wrong dtype on {} has type {}".format(str(node),str(node.dtype)) 
        
        return saturation

            


    def __init__(self, input_shape, delta=0.99, name=None):
        """Init base layer and 'config info' as class variables."""
        super(sat_layer, self).__init__(name=name)
        self.layer_width=input_shape[1:]
        self.layer_width_tensor=tf.constant(input_shape[1:], dtype=tf.float64)
        self.delta = tf.dtypes.cast(delta, dtype=tf.float64)
        #print("Init Satlayer!")
        
    def build(self, input):
        """From shape of input_tensor create initial values of tracked states (keras-weights)
        to be used in reset() and calls it."""        
        input_shape = K.int_shape(input)
        self.inits=[np.zeros(shape=(None)), 
                    np.zeros(shape=input_shape[1:]), 
                    np.zeros(shape=input_shape[1:]*2)]
        
        self.o_s = self.add_weight(shape=(None), name="o_s", dtype=tf.float64, trainable=False)
        self.r_s = self.add_weight(shape=self.inits[1].shape, name="r_s", dtype=tf.float64, trainable=False)       
        self.s_s = self.add_weight(shape=self.inits[2].shape, name="s_s", dtype=tf.float64, trainable=False)
        self.reset()
           
           
         
            
    def update(self, input_tensor):
        """
        Assumes input_tensor has shape (Batchsize X Features) and updates the three
        tracked states given the activation of the parent-layer as input_tensor:
        -Batchsize (scalar) is added to Number of observed samples.
        -Input_tensor summed over batch-dimension (shape F) is added to running_sum (of feature activations?)
        -Sqr_inp acquired by matmuling input_tensor with itself and batch-dim as inner dim. 
            -> gives (shape FxF) matrix that should? have summed over the batch-dim implicitly and is added to sum_squares.
        
        TODO monitor/adjust shapes for different models/inputs esp. leading commas.
        TODO add asserts (at beginning?)
        """
        
        shape=input_tensor.get_shape()#get_shape might fail
        #print("SatLayer.update(): Input tensor has shape: {}".format(shape))
        
        #OBSERVED SAMPLES
        batch_size = shape[0] 
        if not batch_size:
            batch_size=(0.)
        self.o_s.assign_add(batch_size) #add batchsize
        
        #RUNNING SUM
        input_tensor = tf.dtypes.cast(input_tensor, dtype=tf.float64)
        sum_over_batch = tf.reduce_sum(input_tensor, axis=0)
        self.r_s.assign_add(sum_over_batch) #add sum over batch
        
        #SUM OF SQUARES
        sqr_inp = tf.linalg.matmul(input_tensor, input_tensor, transpose_a=True) 
        self.s_s.assign_add(sqr_inp) #add sqr of batch inp
        """((FIRST SQUARE THEN REDUCESUM OVER BATCH: B X F -> B X F X F -> F X F))?
        ->No that happens automatically when squaring the matrix i think ~TODO"""
        
        tf.debugging.assert_shapes([
            (input_tensor, (None, self.layer_width)),
            (self.o_s, ()),
            (self.r_s, (self.layer_width,)),
            (self.s_s, (self.layer_width, self.layer_width)),
            (sum_over_batch, (self.layer_width)),
            (sqr_inp, (self.layer_width, self.layer_width))
        ])
        
        #do i need ctrl-dep to ensure this gets evaluated? assign add shoudl do right?
        return self.o_s, self.r_s, self.s_s 
   
    def reset(self):
        """Reset the running states in the weights to initial value."""
        self.set_weights(self.inits)
        print("RESET Weights!")
        
 
    def __call__(self, input, ):
        """Calls build() if weights/states not initialized. Could be part of update()"""
        if len(self.weights)==0:
            self.build(input)
        return self.update(input)
    
    def result(self):
        """Returns computed saturation based on current states. 
        Function name analogue to keras metric class."""
        return self.sat(self.o_s, self.s_s, self.r_s)
  






