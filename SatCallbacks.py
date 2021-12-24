import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import SatLayer
import importlib
import tensorflow.keras.backend as K
import SatFunctions
import rsc

importlib.reload(SatFunctions)


def layer_summary(self, layer):
    """
        Prints a layers states and results, comparing 
        naive vs. two-pass algorithms. -> cant do that bc tp is tracked in cb TODO remove? 
         
        DEBUGGING ONLY! NOT for saving/logging.
    """
    l = layer
    print(f"\nLayer {l.name} sat_result: {l.sat_layer.result()}")
    if tf.executing_eagerly(): print(f"Observed samples sat_layer: {l.sat_layer.o_s.numpy()}")
    
    layer_values = l.sat_layer.show_states()
    aggr_sat = SatFunctions.get_sat(l.features, *aggr_values, delta=0.99)
    l_sat = l.sat_layer.result()
    l_sat_ag = l.sat_layer.sat(*aggr_values)
    print(f"Sublayer-sat from sublayer_vals: {l_sat}")
    print(f"Sublayer-sat from aggr_vals: {l_sat_ag}")
    print(f"SatFunctions-sat from aggr_vals: {aggr_sat}")
    
    #Compare states
    val_dict = {n : vals for n, vals in zip(l.states, zip(layer_values, aggr_values))}
    for s in l.states:
        rsc.compare_tensors(val_dict[s][0], val_dict[s][1], "Layer_"+s, "Aggregator_"+s)
    

class sat_logger(keras.callbacks.Callback):
    """
        (Messy Callback containing the working/successful loops! and using both algorithm's loops at once! Mamma Mia :D
        Apparently performs 2-pass loop using activations obtained from sat_layer's current_activation weight.
        Also seemingly contains unfinished code snippets meant to implement logging etc later.
        TODO in the future best use this only together with log_layer so the two approaches don't interfere with each other
        
        Original doc below:)    
        
        Callback accessing SatLayer values to log saturation
        and reset SatLayer states (!)
        Logs all batch activations to compute Two-Pass Covariance
        on epoch end.

        TODO add all logging/plotting options in this callback
    """

    def __init__(self, log_targets=None, log_destinations=None, batch_intervall=None):
        
        if log_targets==None:
            log_targets=["SAT"]
        if log_destinations==None:
            log_destinations=["PRINT","CSV"]    
        
        super(sat_logger, self).__init__()
        self.batch_count = 0
        self.batch_intervall = batch_intervall
        self.cb_log = {target: {} for target in log_targets}
        self.log_destinations = dict.fromkeys(log_destinations, None)
        self.batch_log = {}
        self.batch_log_tf = {}
       
    def layer_summary(self, layer):
        """
            TODO CHECK if can be SAFELY DELETED (saw no calls to this), 
            -> probably the non-class layer_summary() has replaced this, but this function has more lines/function
            
            Only documentation was:
            "For Debugging NOT for saving/logging. maybe move to rsc"
        """
        l = layer
        print(f"\nLayer {l.name} sat_result: {l.sat_layer.result()}")
        if tf.executing_eagerly(): print(f"Observed samples sat_layer: {l.sat_layer.o_s.numpy()}")
        for name in l.states: 
            print(f"{name} aggregator: {tf.shape(l.aggregators[name].result())}")
        
        aggrs = [l.aggregators[name] for name in l.states]
        aggr_values = [aggr.result() for aggr in aggrs]
        layer_values = l.sat_layer.show_states()
        aggr_sat = SatFunctions.get_sat(l.features, *aggr_values, delta=0.99)
        l_sat = l.sat_layer.result()
        l_sat_ag = l.sat_layer.sat(*aggr_values)
        print(f"Sublayer-sat from sublayer_vals: {l_sat}")
        print(f"Sublayer-sat from aggr_vals: {l_sat_ag}")
        print(f"SatFunctions-sat from aggr_vals: {aggr_sat}")
        
        #Compare states
        val_dict = {n : vals for n, vals in zip(l.states, zip(layer_values, aggr_values))}
        for s in l.states:
            rsc.compare_tensors(val_dict[s][0], val_dict[s][1], "Layer_"+s, "Aggregator_"+s)

    def on_train_begin(self, logs):
        """
            Initialize lists to log each layers and batches activation.
        """
        for ix, l in enumerate(self.model.layers[1:]):    
          self.batch_log[f"layer_{ix}"] = []
          self.batch_log_tf[f"layer_{ix}"] = []
        
        #CHECK REPLICAS/DISTRIBUTION 
        print(f"cross-replica-ctxt: {tf.distribute.in_cross_replica_context()}")
        print(f"tf.dist.strat:  {tf.distribute.get_strategy()}")   
          
    def on_batch_end(self, batch, logs=None):
        for ix, l in enumerate(self.model.layers[1:]):
            if hasattr(l, 'sat_layer'):
                self.batch_log[f"layer_{ix}"].append(l.sat_layer.current_activation.numpy())
                self.batch_log_tf[f"layer_{ix}"].append(l.log_layer.activation.numpy())
                os, rs, ss, ca = l.sat_layer.get_states()
                #print(f"batch {batch}: o_s: {os}")
        
        return 
        for l in self.model.layers[1:]:
            for target, value_logs in self.cb_log:
                if target == "ACT":
                    self.cb_log["ACT"][f"batch_{self.batch_count}_layer_{l.name}"] = l.sat_layer.current_activation.numpy()
                if target == "STATES":
                    self.cb_log["STATES"][f"batch_{self.batch_count}_layer_{l.name}"] = l.sat_layer.show_states()
                if target == "SAT":
                    self.cb_log["SAT"][f"batch_{self.batch_count}_layer_{l.name}"] = l.sat_layer.result()
                    #TODO Why is this logging and computing an entire sat-result EVERY BATCH??? 
                    # Logging States and Activations can make sense here, but sat? -> maybe only have this on_epoch_end()
       
            #TODO implement own write logs fun to handle different destinations
            #self._write_logs    
        
    
    def log_to_callback(self):    
        """log specified values to callback dict (to json?) for later writing."""
        for l in self.model.layers[1:]:
            for target, value_logs in self.cb_log:
                if target == "ACT":
                    self.cb_log["ACT"][f"batch_{self.batch_count}_layer_{l.name}"] = l.sat_layer.current_activation.numpy()
                if target == "STATES":
                    self.cb_log["STATES"][f"batch_{self.batch_count}_layer_{l.name}"] = l.sat_layer.show_states()
                if target == "SAT":
                    self.cb_log["SAT"][f"batch_{self.batch_count}_layer_{l.name}"] = l.sat_layer.result()
    
    def show_cb_log(self):
        for target, logs in self.cb_log.items():
            for layer_batch_ix, value in logs.items():
                print(f" {target} {layer_batch_ix} : {value}")
                
    def write_cb_log(self):
        pass
        #if dest ==    
       
    def on_epoch_end(self, epoch, logs=None):
        """
            NOTE: -> This one is central, I think the successful Computing and comparing of both algorithm loops happened here?! 
                No description-string here.
        """ 
        
        
        #COMPARE ACTIVATION LOGGING (tf.var vs. keras weight)
        for ix, l in enumerate(self.model.layers[1:]):
            for (batch_k, batch_tf) in zip(self.batch_log[f"layer_{ix}"], self.batch_log_tf[f"layer_{ix}"]): #TODO theres rsc.compare_tensor_lists()
                #print(f"Keras logged activation: {batch_k}\nTF logged activation: {batch_tf}")
                rsc.compare_tensors(batch_k, batch_tf, 'batch_k', 'batch_tf')
                rsc.compare_tensor_lists(batch_k, batch_tf, 'batch_k', 'batch_tf')
        #raise Exception("Just to break before the rest gets printed.")        
         
        #COMPARE RESULTS (Aggregators, sat_layer states, both algorithms) 
        for ix, l in enumerate(self.model.layers[1:]):
            if ix==2:
                return
            
            #TWO-PASS COVMAT
            tp_cov_mat, tp_cov_mat_e = SatFunctions.two_pass_cov(self.batch_log[f"layer_{ix}"])
            #print(f"Two Pass CovMat {l}: {tp_cov_mat}")
            #rsc.compare_tensors(tp_cov_mat, tp_cov_mat_e, "tp_cov_batched", "tp_cov_epoch")
            
            
            #NAIVE COVMAT
            o_s, r_s, s_s, _ = l.sat_layer.get_states()
            naive_cov_mat = SatFunctions.get_cov_mat(o_s, r_s, s_s)
            naive_cov_mat_N = SatFunctions.get_cov_mat(50, r_s, s_s)
            
            rsc.compare_tensors(tp_cov_mat, naive_cov_mat_N)
            
            print(f"o_s: {o_s}")
            
            #COMPARE COV-ALGORITHMS
            print(f"naive_cov_mat.shape: {naive_cov_mat.get_shape()}")
            print(f"rsum_naive: {tf.math.reduce_sum(naive_cov_mat)}, rsum_tp: {tf.math.reduce_sum(tp_cov_mat)}")
            rsc.compare_tensors(tp_cov_mat, naive_cov_mat, "tp_covmat", "naive_cov_mat")
            nevals, nevecs = SatFunctions.get_eig(naive_cov_mat)
            print(f"Eig Naive: {nevals}")
            tpevals, tpevecs = SatFunctions.get_eig(tp_cov_mat)
            print(f"Eig TwoPass: {tpevals}")
            rsc.compare_tensors(tpevals, nevals, "tp_eval", "n_eval")
            print(f"Diff Evals: {nevals - tpevals}")
            
            #RESET STATES 
            self.batch_log[f"layer_{ix}"].clear()    
            l.sat_layer.reset()
            
        
class sat_results(keras.callbacks.Callback):
    """
        (Was this one working well? or did the good stuff happen with the spaghetti sat_logger one? 
        this one probably can't do 2-pass, and looks like other than resetting() it didn't do a lot by itself)
    
        Callback that accesses SatLayer values to print/log saturation
        and reset SatLayer states (!)

        TODO just pass a nested dict as argument for all the options~?
        TODO  
        TODO add all logging/plotting options in this callback
    """

    def __init__(self, log_targets=None, log_destinations=None, batch_intervall=None):
        """
            Didn't have description.
            I see what log_targets/destinations are for, 
            but what was batch_intervall meant to do? Sanity check every nth batch?? #TODO CHECK IF DELETE SAFE (batch_intervall)
        """
        
        if log_targets==None:
            log_targets=["SAT"]
        if log_destinations==None:
            log_destinations=["PRINT","CSV"]    
        
        super(sat_results, self).__init__()
        self.batch_count = 0 
        self.batch_intervall = batch_intervall
        self.log_targets = log_targets
        self.log_destinations = dict.fromkeys(log_destinations, None)
        
        #self._chief_worker_only = True ?
        #if "TB" in log_destinations.keys:
        #    self.log_destinations["TB"] = tf.summary.create_file_writer(logdir="./cloning_eager_dis/logs/tb/sat/")

    
    def on_train_begin(self, logs=None):
        """
            MAYBE IRRELEVANT
        
            Apparently this is only here, because I thought I might need to do something here, but the sat_layer initializes itself.
            If it was only a mental note then -> #TODO CHECK if I can SAFELY DELETE
        """
        pass
        
    def on_batch_end(self, batch, logs=None):
        """
            Didn't have description here.
            
            MAYBE IRRELEVANT
            
            Looks like I was mainly trying/playing around here, like copying current_activation to log-dict! (important elsewhere? where would that be, no)
            #TODO since sat_layer can update by itself, maybe just delete this whole function? CHECK IF SAFE DELETE
            (or rewrite it with the logging once that's planned out)
        """
        for l in self.model.layers[1:]:
            if hasattr(l, 'sat_layer'):
                logs[f'{l.name}_act'] = l.sat_layer.current_activation.numpy()
                print(f"current_activation: {l.sat_layer.current_activation.numpy()}")
                
                #for target in self.log_targets:
                #if "ACT" in self.log_targets:
                    #logs.update(l.name + '_act_' + str(self.batch_count) : l.sat_layer.current_activation.numpy())
                
                #TODO implement own write logs fun to handle different destinations
                #self._write_logs    
        
    def on_epoch_end(self, epoch, logs=None):
        """
            Didn't have description here.
            I think this doesn't call sat_layer.sat() because rsc.layer_summary() did? probably
            
            sat_layer.reset() is important, but otherwise, not much functionality in this CB hmm
        """
        
        for l in self.model.layers[1:]:
            if hasattr(l, 'sat_layer'):
                    
                #for dest in self.log_destinations:
                    
                
                #rsc.layer_summary(l)
                l.sat_layer.reset()
                
                #for s in l.states:
                #    l.aggregators[s].reset_state()        
                
                
                
                
                
class sat_callback(keras.callbacks.Callback):
    """
        Callback retrieves 'raw' activations exposed in [log_layer] instances 
        and processes activations further according to selected cov-algorithm.
        
        
        -> outsource computations to rsc.
        -> take just the activations from the sublayers 
        (which one? using sat_layer.current_activation? -> i guess access with l.get_weights() ) 
        -> evaluating transient tensors in callback seems possible if one adds dummy outputs, but thats gonna convolute everything man
        
        TODO Decisions: dynamic sized tensor list for 2PASS: 
            (1.) RaggedTensor  (supposedly as the 'tf'-dynamic list, but meh, don't really need the ragged dim either)  
        ->  2. .numpy()      (seems like it would dodge any tf side-effects)
            4. tf.concat / dynamic tensor
            5. logs dict     (worked before, but not elegant or good or robust probably)
            (6.) TensorArray  (I think specific for 'while_loop' iteration, who knows what tf will do if it's called once per callback)       
             
            https://www.tensorflow.org/guide/effective_tf2#do_not_keep_tftensors_in_your_objects  
            -> tensor objects behave differently in tf.function (graph~) vs. eager ctxt. Only for intermediate values.
            To track state, use tf.variables, they're always usable from both ctxts, (can't be reshaped / not dynamic)
            -> problems with both tensors and variables -> use numpy to track 
            -> rn just do it simple and list np.array(tensor.numpy()) and then np.concatenate(list) (don't need stack() for 2P Ithink)
    """
    
    
    def __init__(self, log_targets=None, log_destinations=None, batch_intervall=None, cov_alg=None, instruction_dict=None, **kwargs):
        """
            Initializes attributes etc. depending on specified mode.
            batch_intervall is copied from sat_results, it's really unnecessary isn't it?
            Should have added args or arg dict to specify behaviour/mode. instruction_dict or sth or just kwargs?
            
            cov_alg = "naive", "2pass", ?"both"? or rather use if naive: track naive, if 2pass: track that too,...?
            epoch_intervall = x epochs (to skip before measuring sat again)
            log_targets: "ACT" i guess, "SAT", 
            
            
            #TODO wtf is batch_intervall an arg? cpy pasted? delete if safe. -> epoch intervall only. (+decide if instr_dict is an arg)
            
        """                
        super().__init__(**kwargs)
        
        # separate flags for toggling tracking of either algorithm
        self.naive = instruction_dict["track_naive"] #just use separate keys to check if one algorithm should be tracked
        self.two_pass = instruction_dict["track_two_pass"]
        
        self.epoch_interval = instruction_dict["epoch_interval"]
        
        
        if self.naive:
            self.naive_states = {'o_s': 0, 'r_s': np.zeros(...), 's_s':""}
    
    
    
    def on_begin_fun(self):
        """
            Initialize states to track layers for both algorithms. 
            Called by on_train_begin(), on_predict_begin()
            
            2PASS:
            NAIVE:
        """
        
        
        # NAIVE:
        self.layer_naive_t_s = {}
        for l in self.model.layers[1:]:            
            if hasattr(l, 'log_layer'):
                self.layer_naive_states[l.name] = {
                    'o_s': np.as_array((0.)),
                    'r_s': [], #TODO inject shape here -> shape as arg? (might also query l.width() or sth, since batch collapses)
                    's_s': []
                } 
        
        # 2-PASS:
        self.layer_activations = {}
        for l in self.model.layers[1:]:            
            if hasattr(l, 'log_layer'):
                self.layer_activations[l.name] = [] 
        
    
    def on_epoch_begin(self, epoch, logs=None):
        """
            Check if this epoch should be used for saturation. 
            (Toggle active control flag)
        """
        
        # Active only once per epoch interval in training case (add offset to start later after first y epochs?)
        if epoch%self.epoch_interval == 0:
            self.active = True
        else:
            self.active = False    
            return  #in case I want to add code for beginning active epochs after this block
    
    def on_epoch_end(self, epoch, logs=None):
        
        # Epoch activity:
        if not self.active:
            return
        else:
            self.active = False
        
        
        
        if self.naive:
            cov_mat_naive = rsc.cov_mat_naive()#from tracked states passed
        
        
    def on_batch_end(self, batch, logs=None):
        """

        """
        
        if not self.active:
            return
        
        # NAIVE:
        
        
        # 2-PASS: 
        for layer_name in self.layer_activations:
            act = self.model.get_layer(layer_name).log_layer.activation.numpy()
            self.layer_activations[layer_name].append(act) 
            print(f"Batch {batch} Layer {layer_name} Activation: \n {act}\n")
                
            