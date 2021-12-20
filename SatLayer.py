import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow import keras
import numpy as np

"""
    Classes (extending keras.Layer) used to measure FSS of a "parent" keras Layer
    by attaching to "parent"-layer as Sublayer. 


    Functionally similar to stateful keras metrics relevant states are 
    updated in update(), reset in reset() and queried with result().
    The core computation of Saturation given the states is done in sat(). 

    based on SatMod_modstates_current.py 15.10.2020 2:21
    to be used by Clone_Model extended Layers as sublayer

"""


class sat_layer(keras.layers.Layer):
    """
        Sublayer of mydense layer to measure FSS and related Metrics,
        by updating three states based on activation of "parent"-layer's (batch-)activation each forward-pass
        ...
        
        ATTRIBUTES
        ---------- 
            TRACKING ALGORITHM STATES:
                o_s : keras Weight 
                    [scalar] Number of observed samples 
                r_s : keras Weight
                    [Features] Running Sum of activations (NOT Running Mean, that's only computed when needed in sat())
                s_s : keras Weight 
                    [Features X Features] Sum of Squares of activations
                current_activation : keras Weight
                    Is assigned each batches activation(/input)-tensor 
                    for logging/saving intermediate activations with model.save / Checkpoint callback               
                inits : list
                    zero values to initialize and reset layer weights (states) with     
                
            GENERAL/LAYER PROPERTIES:    
                delta : tensorflow Variable/Node
                    [scalar] delta value used as threshold of ratio of used Eigenvalues to total Variance
                layer_width(_tensor) : str (or tf.constant)
                    number of units of (dense) Parent-layer. 
                    
        METHODS
        -------
            RUNNING + BUILDING: (core of loop with sat_results Callback)
                __call__(input)
                    For each batch/forward pass call update() and when weights not created build(); 
                    can add metrics (eg.logging states) here ~>meh with the accumulation   
                build(input)    
                    create state-weights and inits then calls reset() to initialize them
                update(input_tensor)        <- #TODO REVIEW: overlaps with functions of other/modular approach
                    computes update-values from input_tensor and adds them to state-weights
                
                                            
            RUNNING/UPDATING ALTERNATIVES: 
            -> Unused rn i think, maybe in callbacks, but cleaner approach? #TODO REVIEW: Decide on approach!! Clean up!   
                get_update_values(input_tensor)
                    compute and return update-values from input_tensor without sideeffects (updating states)
                update_state(batch_size, sum_over_batch, sqr_inp)
                    update state-weights with precomputed update-values
                call(self, input)
                    not used, not needed I think ~
    
                
                 
            COMPUTING/ACCESSING RESULT:    
            -> I think only called as core loop from sat_results.end_of_epoch() CB
                result()
                    returns value for saturation by calling sat() with current state of state-weights
                sat(o_s, runn_sum, sum_sqrs)
                    compute and return a saturation from the given states
                lossless_k_loopless(evals)
                    from a list of eigenvalues (largest first) finds number k of corresponding eigenvectors 
                    necessary to span a ~~~lossless subspace~~~ without iterating over the list to avoid a branching tf graph.          
                ll_k_fun(evals)
                    wraps lossless_k_loopless() as a tf.function            
                reset()
                    sets the three state-weights to zero using inits (called from external callback)    


            ACCESSING FROM CALLBACK GENERAL:        
            -> Used from sat_logger CB to essentially perform both algorithms externally using the state-values
                get_states()
                    returns (separately) the tracked states including self.current_activation
                show_states()
                    returns list of current values of (the 3 core) state-weights (external query to observe and check process)
                        
        TODO check if this should have set dynamic=True because i manipulate tensors with python?    
        TODO probably just outsource all non-stateful functions to SatFunctions.py     
        TODO check for abnormally sized batches and either skip or find solution for the dim-change    
        TODO wrap get_update_values(), update_state() into update() to use independent of classes updates (eg.sanity testing)               
        TODO add definition of layer_width for conv version of class  
        TODO debug functionality! @tf.function ll_k()? -> test both options! correct values? 
        TODO make process efficient (GPU, DistributeStrategy, no internal tf/keras bottlenecks -> tf.Profiler)    
        TODO ensure states are tf.Strategy.reduce()d (sum) over all replicas before computing saturation when training distributetly
    """
    
    def lossless_k_loopless(self, evals):
        """
            Determines minimal number k (+1) of largest evals neccessary for spanning a 
            variance eigenspace of delta.
            k := argmax(sum(k_largest_evals)/N) <= delta
            
            {"list"(tensor) of evals -> [sort] -> [cumsum] cumulative evals 
            -> [divide] by 'N' (covmat-trace/layer variance) -> [var_ratio_evs<=delta] cond (1/0 booleans) 
            -> [rsum] (scalar) number cum_evs<=d -> k   [+track corresponding ixs]}
            '<=' is checked for all list-ixs (avoids looping in the Graph) at once, so k is found 
            by rsum over (boolean True/1) elems of the bool-list cond. (less hacky than it looks)
            
            PARAMETERS:
                evals (?list / tf.tensor?): listed eigenvalues of the Covariance-matrix used
            
            RETURNS: 
                k (tf.tensor):          resulting k (scalar)
                k_s_ixs (tf.tensor?):   indices of the k largest Eigenvalues / used Principal Components 
        """
        #assert evals.dtype == tf.float64, "EigVals not f64 in llk"
        
        #Sort in DESC. Order 
        s_evs = tf.sort(evals, axis=0, direction='DESCENDING', name="sort_ev") 
        s_ixs = tf.argsort(evals, axis=0, direction='DESCENDING', stable=True, name="sort_ev_ix")  
        #print("evals: {}\nevals_sorted_desc: {}".format(evals, s_evs)) 
        
        #Get list of cumulative sum of largest EVs
        cum_evs = tf.cumsum(s_evs, axis=0, name="cumsum_evals") 
        #print("Cumulating EVs: {}".format(cum_evs))
        
        #Divide Cum.sums by total sum of EVs (trace(CovMat) ~> variance of CovMat)
        dim_ratios = tf.realdiv(cum_evs, cum_evs[-1], name="div_by_trace")
        #print("Dim ratios: {}".format(dim_ratios))
        
        cond = tf.math.less_equal(dim_ratios, self.delta) #TODO change first F ix
        #print("cond: {}".format(cond))
        
        k_s_ixs = tf.boolean_mask(s_ixs, cond)
        
        k = tf.math.maximum(tf.dtypes.cast(len(tf.where(cond)), dtype=tf.float64) , tf.constant([1.], dtype=tf.float64))#tf.reduce_sum(tf.where(cond), axis=-1, name="k")
         
        return k, k_s_ixs

    @tf.function
    def ll_k_fun(self, evals):
        #TODO check if neccessary
        return self.lossless_k_loopless(evals)        
    
    def sat(self, o_s, runn_sum, sum_sqrs):
        """
            
            Computes Saturation from values for the three naive Algorithm states. 
            Doesn't depend on or affect SatLayer states/members !other than self.delta in ll_k()!
            
            NAIVE ALGORITHM
            
            CALLS:      ---    
            CALLED BY:  self.result()
            
            PARAMETERS:
            
                o_s (scalar tf.Tensor / tf.Variable / keras Weight):  
                    Tensor of shape (1,) containing the scalar count of observed samples 
                
                runn_sum (tf.Tensor / tf.Variable / keras Weight): 
                    Tensor of shape (F) containing the running sums of activations of single units/features
                    
                sum_sqrs (tf.Tensor / tf.Variable / keras Weight): 
                    Tensor of shape (FxF) containing the sums of products of pairs of 
                    features/units(' column/row vectors of activation-tensor along the batch-dimension) 
                    
                    That is equivalent to summing the 'o_s' many squares 'shape(F X F)' of all observed
                    feature-activations/activation-vectors 'shape(F)' for single samples all at once 
                    as if they were "one batch".
                    The Squaring-and-then-Summing of of all the single-samples' feature-activations is only
                    split up and performed "batchwise" in the form of matrix-squaring (matmul w itself) 
                    the activation-tensor of the minibatch so the batch dimension is the inner one getting 
                    reduced, (F X B) o (B X F) resulting in a (F X F) tensor where each element (i,j) 
                    is the sum of all products of scalar (single-sample, single-)feature activations 
                    of the features i,j over all samples in the batch.
                    Computing the Sum of Squares as a sum of sub-solutions sequentially while training 
                    on each batch requires neither that all squares be loaded into memory at once or centrally
                    nor to transform the form the data has before applying an operation, which in this case
                    is one that can be naturally performed by the same hardware as the training (is taking place on?)
                    (Intermediate solutions that remain in memory to be used further eg. combining/reducing is dynamic programming right?)
                
            RETURNS:
            
                saturation (scalar tf.tensor):      saturation given inputs (current states) and self.delta
                
            TODO use separate SatFunctions  (e.g. cov_mat() function from rsc!! cleaner)
            TODO after tinkering outsource to SatFunctions? since it is static
        """    
        
        #SAMPLE MEAN (from running sum of features and running sample count) (Not computed every batch)
        sample_mean = tf.realdiv(runn_sum, o_s, name='sample_mean') #alt tf.math.scalar_mul()
        
        #DOT PRODUCT OF SAMPLE MEAN
        sample_mean_dot = tf.tensordot(sample_mean, sample_mean, axes=0, name='sample_mean_dot') #alt tf.einsum()

        #COVARIANCE MATRIX
        cov_mat = tf.math.subtract(tf.realdiv(sum_sqrs, o_s), sample_mean_dot, name='cov_matrix') 
        
        #EIGEN DECOMPOSITION TODO invert order here?, TODO check eig() as possible error source -> eigh? 
        eig_vals_raw, eig_vecs_raw = tf.linalg.eigh(cov_mat, name='eig_decomp') 
        eig_vals = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vals_raw)), dtype=tf.float64, name="eig_vals_real")
        eig_vecs = tf.cast(tf.math.real(tf.convert_to_tensor(eig_vecs_raw)), dtype=tf.float64, name="eig_vecs_real")#why convert_to_tensor again? eig shld rtrn that 
        
        #K
        if tf.executing_eagerly(): #still needed?
            k, ixs = self.lossless_k_loopless(eig_vals) #abs.val compl ev #tf.constant(4, dtype=tf.int32)
        else:
            k, ixs = self.ll_k_fun(eig_vals)
        #print("K tensor: {}".format(k))

        #PROJECTION MATRIX  TODO alt. tf.slice + tune indices (also for batches)
        k_eig_vecs = tf.gather(params=eig_vecs, indices=ixs , axis=0, batch_dims=1, name="k_eig_vecs") 
        proj_mat = tf.matmul(k_eig_vecs, k_eig_vecs, transpose_a=True, name="proj_mat")

        saturation = tf.truediv(k, self.layer_width_tensor, name="saturation")
        
        #ASSERTS dtypes
        for node in [sample_mean,
                     sample_mean_dot, cov_mat, eig_vals_raw, eig_vals,
                     eig_vecs, k, k_eig_vecs, proj_mat, saturation]:
            assert node.dtype in [tf.float64, tf.complex128], "Wrong dtype on {} has type {}".format(str(node),str(node.dtype)) 
        
        return saturation

    def __init__(self, input_shape, delta=0.99, name=None, cov_alg="naive"):
        """
            Initializes basic layer and "config information" as class variables.
            
            CALLS: keras.layers.Layer/super(...).__init__()
            CALLED BY: mydense.process_params()
            
            PARAMETERS:
                input_shape (tuple?): shape of "parent"-layer's activation
                delta (float): set threshold
                name (str): set name to access this sublayer with
                cov_alg (str): Algorithm chosen to estimate Covariance-matrix
        """
        
        super(sat_layer, self).__init__(name=name) #TODO ?? self = keras.layers.Layer.__init__() ??
        self.layer_width=input_shape[1:]
        self.layer_width_tensor=tf.constant(input_shape[1:], dtype=tf.float64)
        self.delta = tf.dtypes.cast(delta, dtype=tf.float64)
        self.act_shape=()
        self.cov_alg=cov_alg
        
    def build(self, input_shape):
        """
        
            Creates keras weights / np arrays to contain tracked states, init-values, 
            and current_activation to match given input_shape. 
            Then uses reset() to set to intial values. 
            
            NAIVE ALGORITHM
            
            CALLS:      (self.reset())
            CALLED BY:  self.__call__()
            
            TODO pass input_shape here
            TODO use layer_width for init shapes instead of the input-tensor
            TODO delete current_activation in case it's not used in the naive updating process (may be used for debugging from callback)
            
            PARAMETERS:
                input_shape (tuple):    shape of "parent"-layer activation
                    
        """        
        n_spls_epoch=0
        #input_shape = K.int_shape(input)
        print(f"input_shape: {input_shape}")
        #print(f"input.get_shape(): {input.get_shape().as_list()}")
        print(f"layer_width: {self.layer_width}")
        #maybe need to use np.dtype==float64 hier schon sonst wird das beim casten ungenau! TODO
        #fehler von o_s ist pro layer zwischen batches gleich! dh. die initwerte sind das Problem!!
        self.inits=[np.zeros(shape=(1,), dtype=np.float64), 
                    np.zeros(shape=input_shape[1:]), 
                    np.zeros(shape=input_shape[1:]*2),
                    np.zeros(shape=(0, *self.layer_width)),
                    np.zeros(shape=(n_spls_epoch, *self.layer_width))] #TODO see how the batchsize 10 here can be made variable otherwise you need to know and hardcode batchsize in advance
                    #either pass the batchsize :/, somehow make a list
                    #or use a tensor containing all batches/samples and guess/pass number of samples beforehand. 
                    #every pass we could insert a new batch taking the place of zeros or NAs


                    
        self.o_s = self.add_weight(shape=self.inits[0].shape, name="o_s", initializer='zeros', dtype=tf.float64, trainable=False)
        self.r_s = self.add_weight(shape=self.inits[1].shape, name="r_s", initializer='zeros', dtype=tf.float64, trainable=False)       
        self.s_s = self.add_weight(shape=self.inits[2].shape, name="s_s", initializer='zeros', dtype=tf.float64, trainable=False)
        self.current_activation = self.add_weight(shape=self.inits[3].shape, name="curr_act", initializer='zeros', dtype=tf.float64, trainable=False)
        self.epoch_activations = self.add_weight(shape=self.inits[3].shape, name="epoch_act", initializer='zeros', dtype=tf.float64, trainable=False)
        
        self.built = True
        self.reset()
    
    def get_update_values(self, input_tensor):
        """
            COMPUTES update values WITHOUT APPLYING CHANGES to the states. 
            Mostly for Metric approach
            
            NAIVE ALGORITHM
            
            CURRENTLY NOT USED
            
            PARAMETERS:
                input_tensor (keras tensor): activation of "parent"-layer
                
            RETURNS:
                batch_size, sum_over_batch, sqr_inp (tf tensors):   Values to update the states by (by adding)    
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
    
    def update_state(self, batch_size, sum_over_batch, sqr_inp):
        """
            Updates states in the sat_layer by given values obtained elsewhere.
            
            NAIVE ALGORITHM
            
            CURRENTLY NOT USED    
            #TODO + Modularity! separate the updating, generating, ... 
            
            
            PARAMETERS:
                batch_size (tf.tensor):         scalar increase in observed samples with newest batch -> o_s
                sum_over_batch (tf.tensor):     sum over activations in newest batch -> r_s
                sqr_inp (tf.tensor):            square of batch-activation -> s_s
                
            RETURNS:
                self.o_s, self.r_s, self.s_s (references to class members?): I need to return something, in order for the assigns to be executed by tf as a graph dependency right?
            
        """
        self.o_s.assign_add(batch_size)
        self.r_s.assign_add(sum_over_batch)
        self.s_s.assign_add(sqr_inp)
        
        #TODO return assign operations here? no i think returning a callable is meant for metrics
        return self.o_s, self.r_s, self.s_s 
                    
    def update(self, input_tensor):
        """

            NAIVE ALGORITHM
            
            COMPUTES and UPDATES the sat_layer's three states for the naive Algorithm
            (update values obtained from the layer-activation input_tensor).
            Also returns the update values? / assign operations???.
            
            CALLS:      --- (not using the other functions atm)
            CALLED BY:  self.__call__()
            
            Updates:    
                Observed Samples (Counter) += Batchsize (scalar)
                Running Sum (of features/activations) += Input_tensor summed over batch-dim (shape F)
                Sum of Squares +=  Squared input (matmul input_tensor with itself, batch-dim as inner dim) (shape FxF) 
                    (-> should have summed over the batch-dim implicitly~?.)
                
            PARAMETERS:
                input_tensor (keras Data?Tensor): to receive the (Batchsize X Features) activation of the "Parent"-layer

            RETURNS:
                self.o_s,r_s,s_s (?): 
                
                #TODO returning self.state here might return references instead of values. 
                #Was this a hack? Do I need to do this so tf executes it in the Graph? Does this return the graph nodes of assign_add()?
                
            
            Soon Replaced by update_state() and get_update_values()? Wrap them?~ better leave this working.
            TODO monitor/adjust shapes for different models/inputs esp. leading commas.
            TODO raise Error when input_tensor has wrong shape / or pass for dry runs?
            TODO catch smaller batches when saving/assigning the batch-activation by padding with NA?
            -> or use list that creates/appends a new variable, everytime a new batchsize appears? no priority 
        """
        input_tensor = tf.dtypes.cast(input_tensor, tf.float64)    
        shape=input_tensor.get_shape() #get_shape might fail
        
        #print(f"input_tensor: {input_tensor}")
        #print(f"shape: {shape}")
        #print(f"batchsize: {shape[0]}")
        #print(f"o_s: {self.o_s.shape}")
        
        #OBSERVED SAMPLES
        batch_size = [shape[0]] 
        if batch_size == [None]:
            batch_size = [0.] #TODO here we would have to use the specified batch_size passed in satify_model()
            print(f"batch_size initialized. current shape is: {tf.shape(input_tensor[0])}")
        self.o_s.assign_add(batch_size) 
        
        #RUNNING SUM
        sum_over_batch = tf.reduce_sum(input_tensor, axis=0)
        self.r_s.assign_add(sum_over_batch)
        
        #SUM OF SQUARES
        sqr_inp = tf.linalg.matmul(input_tensor, input_tensor, transpose_a=True)
        self.s_s.assign_add(sqr_inp)
        """((FIRST SQUARE THEN REDUCESUM OVER BATCH: B X F -> B X F X F -> F X F))?
        ->No that happens automatically when squaring the matrix i think ~TODO"""
        
        #LOGGING THE BATCH-ACTIVATION
        if batch_size != [None] and self.cov_alg=="two-pass":
            self.current_activation.assign(input_tensor) #TODO breaks here because batchsize static
                
        #LOGGING BATCH-ACTIVATION IN EPOCH-SPANNING TENSOR
                
                
        tf.debugging.assert_shapes([
            (input_tensor, (None, self.layer_width)),
            (self.o_s, ()),
            (self.r_s, (self.layer_width,)),
            (self.s_s, (self.layer_width, self.layer_width)),
            (sum_over_batch, (self.layer_width)),
            (sqr_inp, (self.layer_width, self.layer_width))
        ])
        
        return self.o_s, self.r_s, self.s_s 
   
    def reset(self):
        """
            Resets the weights containing tracked states to initial values.
            
            NAIVE ALGORITHM
            
            CALLED BY:      self.build(), Callback/sat_results.on_epoch_end()
            
            https://github.com/keras-team/keras/issues/341  <- reset keras weights (very different)
            TODO broadcast to all replicas when distributed.
        """
        self.set_weights(self.inits)
        
    def __call__(self, input, ):
        """
            Passes input/activation to self.update() and returns resultS 
            so "parent"-layer depends on execution of FSS updates. (necessary?)
            Calls build() if weights/states not initialized yet. 
            
            NAIVE ALGORITHM
            
            CALLS:      self.update(input),     self.build()
            CALLED BY:  "parent"/mydense.call()
            
            PARAMETERS: 
                input (keras tensor): activation passed by "parent"-layer
                
            RETURNS:
                update_results (tf tensors): outputs of self.update(input)  
            
            TODO Merge with update()? ~wrapping it like now is readable and modular
        """
        
        #self.add_metric(self.o_s, name="o_s_sl_metric", aggregation='mean')
        
        #BUILD LAYER ON FIRST TIME
        if len(self.weights)==0:
            self.build(input.get_shape().as_list()) #TODO get shape right here
            
        #CHECK IF INPUT-SHAPE MATCHES FIRST INPUT    #TODO catch wrong-sized batches here!!! -> just add empty return ?
        if input.get_shape() != self.current_activation.get_shape():
            print(f"SatLayer input shape {input.get_shape()} not matching current_activation property shape {self.current_activation.get_shape()}")    
        
            
        return self.update(input)
    
    def call(self, inputs, training=None):
        """
            CURRENTLY NOT USED ?
            
            
            I think this was meant as part of returning a callable to "parent"-layer.call(), maybe through __call__(),
            while providing DIFFERENT BEHAVIOUR depending on whether model is TRAINING and possibly other cases like update-strategy...! 
            
            Only documentation was:
            "this should be called by __call__() (egal weil kein gradient wrsl) and can check if training."
            
            CALLS:      self.update()
            CALLED BY:  ---
            
            PARAMETERS:
                inputs (keras tensor):  "parent"-layer activation
                training (Bool?/str?):  flag indicating if model is training    
            
            RETURNS:
                results of self.update() (again probably meaning to ensure execution with graph dependency)
                  
            
        """
        if training:
            return self.update(inputs)
        return
            
    def result(self):
        """
            Returns computed saturation based on current states.
            (Access-point for fetching result from callback)
            
            CURRENTLY USED: CALLBACK ACCESS (lol actually from sat_logger, and not from sat_results)
            
            NAIVE ALGORITHM
            
            CALLS:      self.sat()
            CALLED BY:  layer_summary() <- sat_logger.on_epoch_end() (Result accessed from CB)  
                                            (I thought for access from sat_results, but there it's not actually used rn)
            
            RETURNS:    
                saturation (scalar tf.tensor):  saturation based on current states

            
        """
        return self.sat(self.o_s, self.r_s, self.s_s)
  
    def show_states(self):
        """
            Returns current state for external access from callback/metric. 
            (for logging, debugging or inspection) -> probably only debugging!
            
            CURRENTLY USED: EXTERNAL (CALLBACK) ACCESS, (DEBUGGING ONLY??)
            
            NAIVE ALGORITHM
            
            CALLED BY:  layer_summary() <- sat_results.on_epoch_end()   (States accessed from Callback: DEBUGGING)
                        sat_logger.on_batch_end()                       (Callback: apparently LOGGING, though maybe logging this is for debugging/inspecting itself?)
                        sat_logger.log_to_callback()
                        
            RETURNS:
                states (list):  list of current states (values or nodes?)
            
        """
        return [self.o_s, self.r_s, self.s_s]

    def get_states(self):
        """
            Returns current states for external access from callback. 
            (for logging, debugging or inspection) -> like show_states() -> probably only debugging!
            
            CURRENTLY USED: EXTERNAL CALLBACK ACCESS (DEBUGGING ONLY??)
            
            NAIVE ALGORITHM + 2-PASS ALGORITHM
            
            CALLED BY:  sat_logger.on_epoch_end()
            
            RETURNS:
                states (multiple):  separately returned current states (values or nodes?)
            
        """
        
        return self.o_s, self.r_s, self.s_s, self.current_activation

class log_layer(keras.layers.Layer):
    """
        NOTE: Looks like this was actually an attempt at logging from the layer :/ 
        Bad Idea, logging functionality should be entirely in CBs (to avoid spaghetti), 
        while in the "external" path approach, the sublayer needs only to track and make accessible states!
        The further processing (FSS computation from states, logging of states/results) should happen entirely in CB
        -> So I need a tracking_layer instead of a log_layer !  
        
        #TODO turn this into tracking/exposing layer
        #TODO decide: expose as keras weight or tf.variable? init in __init__() or build()?
        
        EXPOSES activations for CB access. (without casting dtype).
        
        
    """
    
    
    def __init__(self, input_shape, dtype=None, name=None, **kwargs):
        """
            Initialize a basic Layer object. (maybe the activation var too, but probably rather in build())
            
        """
        
        
        super(log_layer, self).__init__(name=name, **kwargs)
        
        #maybe just not do this here?
        """
        self.activation = tf.Variable(shape=input_shape,
                                      initial_value=np.zeros(shape=(1,)+shape[1:],dtype=np.float64),
                                      validate_shape=False,
                                      dtype=dtype,
                                      trainable=False)
        """
        
    def build(self, input_shape):
        """
            build(self, input_shape) should be called by __call__() on first call, I guess __call__() will get and pass input_shape~?
            
            
            https://keras.io/guides/making_new_layers_and_models_via_subclassing/#best-practice-deferring-weight-creation-until-the-shape-of-the-inputs-is-known
            -> also maybe var creation needs to happen in a tf.init_scope() ? TODO
        
        """
        
        #work in progress
        self.activation = tf.Variable(shape=input_shape, 
                                      validate_shape=False,
                                      trainable=False)
        
        #No idea where I copy pasted this from but probably safe to delete other than the shape stuff
        """
        self.activation = tf.Variable(#shape=(1)+input_shape[1:],
                                      initial_value=np.zeros(shape=[1]),
                                      dtype=tf.float64,
                                      validate_shape=False,
                                      trainable=False)
                                      #TensorSpec=(None))
        #self.activation.assign(np.zeros())
        """
        
    def update(self, activation):
        """ 
            Assigns the new activation to the tracking variable.
        """
        
        self.activation.assign(activation) #, validate_shape=False)
    
        return self.activation 
   
    
    def call(self, activation):
        """
            Returns update function that then assigns the activation batch-tensor.
        """
        return self.update(activation)
    
    #ideally not overwrite that function
    #def __call__(self, input):