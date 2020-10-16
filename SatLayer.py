import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow import keras
import numpy as np

"""
based on SatMod_modstates_current.py 15.10.2020 2:21
to be used by Clone_Model extended Layers as sublayer

"""



class sat_layer(keras.layers.Layer):

    def lossless_k_loopless(self, evals):
        assert evals.dtype == tf.float64, "EigVals not f64 in llk"
        s_evs = tf.sort(evals, axis=0, direction='ASCENDING', name="sort_ev")#necessary?
        s_ixs = tf.argsort(evals, axis=0, direction='ASCENDING', stable=True, name="sort_ev_ix") 
        n_s_evs = tf.realdiv(s_evs, self.obs_samples, name="norm_evals_by_os")
        sum_n_s_evs = tf.cumsum(n_s_evs, axis=0, name="cumsum_evals")
        cond = tf.math.less_equal(sum_n_s_evs, self.delta) #TODO change first F ix
        k = tf.reduce_sum(tf.where(cond), axis=0, name="k_sketch")
        k_s_ixs = tf.boolean_mask(s_ixs, cond)
        return k, k_s_ixs


    #dont do updating here only get sat(current states)
    def sat(self, o_s, sum_sqrs, runn_sum):
        
        #running mean is probably cleaner to get adhoc from current o_s and r_s right? is that the same?
        sample_mean = tf.realdiv(runn_sum, o_s, name='sample_mean') #alt tf.math.scalar_mul()

        sample_mean_dot = tf.tensordot(runn_sum, runn_sum, axes=0, name='sample_mean_dot') #alt tf.einsum()

        cov_mat = tf.math.subtract(tf.realdiv(sum_sqrs, o_s), sample_mean_dot, name='cov_matrix') 

        eig_vals_raw, eig_vecs_raw = tf.linalg.eig(cov_mat, name='eig_decomp') 
        
        eig_vals = tf.cast(tf.math.abs(tf.convert_to_tensor(eig_vals_raw)), dtype=tf.float64, name="eig_vals_abs")
        
        eig_vecs = tf.cast(tf.math.abs(tf.convert_to_tensor(eig_vecs_raw)), dtype=tf.float64, name="eig_vecs_abs")#why convert_to_tensor again? eig shld rtrn that 

        k, ixs = self.lossless_k_loopless(tf.math.abs(eig_vals)) #abs.val compl ev #tf.constant(4, dtype=tf.int32)

        k_fl = tf.cast(k, dtype=tf.float64, name="k_fl")

        k_eig_vecs = tf.gather(params=eig_vecs, indices=ixs , axis=0, batch_dims=1, name="k_eig_vecs") #TODO or tf.slice, tune indices (also for batches)
        
        proj_mat = tf.matmul(k_eig_vecs, k_eig_vecs, transpose_a=True, name="proj_mat")

        saturation = tf.truediv(k_fl, self.layer_width, name="saturation")
        
       
        
        for node in [inp, sqr_inp, sum_sqrs, running_sum, sample_mean,
                     sample_mean_dot, cov_mat, eig_vals_raw, eig_vals,
                     eig_vecs, k_fl, k_eig_vecs, proj_mat, saturation]:
            assert node.dtype in [tf.float64, tf.complex128], "Wrong dtype on {} has type {}".format(str(node),str(node.dtype)) 
        
        return saturation

            


    def __init__(self, input_shape, delta=0.99, name=None):
        super(sat_layer, self).__init__(name=name)
        
        self.layer_width=input_shape[1:]
        self.delta = tf.dtypes.cast(delta, dtype=tf.float64)
        
    def build(self, input):
        
        input_shape = K.int_shape(input)
        self.inits=[np.zeros(shape=(None)), 
                    np.zeros(shape=input_shape[1:]), 
                    np.zeros(shape=input_shape[1:]*2)]
        
        self.o_s = self.add_weight(shape=(None), name="o_s", dtype=tf.float64, trainable=False)
        self.r_s = self.add_weight(shape=self.inits[1].shape, name="r_s", dtype=tf.float64, trainable=False)       
        self.s_s = self.add_weight(shape=self.inits[2].shape, name="s_s", dtype=tf.float64, trainable=False)
        self.reset()
           
           
         
            
    def update(self, input_tensor):
        
        shape=input_tensor.get_shape()#hm get_shape might fail
        
        batch_size = shape[0] 
        if not batch_size:
            batch_size=(0.)
        
        input_tensor = tf.dtypes.cast(input_tensor, dtype=tf.float64)
        
        sum_over_batch = tf.reduce_sum(input_tensor, axis=0)
        
        #FIRST SQUARE THEN REDUCESUM OVER BATCH 
        #B X F -> B X F X F -> F X F
        #No that happens automatically when squaring the matrix right?
        sqr_inp = tf.linalg.matmul(input_tensor, input_tensor, transpose_a=True) 

        
        self.o_s.assign_add(batch_size) #add batchsize
        self.r_s.assign_add(sum_over_batch) #add sum over batch
        self.s_s.assign_add(sqr_inp) #add sqr of batch inp
        
        return self.o_s, self.r_s, self.s_s #do i need ctr deps to ensure this gets evaluated? assign add shoudldo
        
   
    def reset(self):
        self.set_weights(self.inits)
        pass
        
 
    def __call__(self, input):
        if len(self.weights)==0:
            self.build(input)
        return self.update(input)
        #return self.saturation
            
  






