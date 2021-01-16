import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import SatLayer
import importlib
import tensorflow.keras.backend as K
import SatFunctions
import rsc

importlib.reload(SatFunctions)


class sat_logger(keras.callbacks.Callback):
    """
    Callback that accesses SatLayer values to log saturation
    and reset SatLayer states (!)
    Logs all batch activations to compute Two-Pass Covariance
    on epoch end.

    TODO add all logging/plotting options in this callback
    """

    def __init__(self, log_targets=["SAT"], log_destinations=["PRINT","CSV"], batch_intervall=None):
        super(sat_logger, self).__init__()
        self.batch_count = 0
        self.batch_intervall = batch_intervall
        self.cb_log = {target: {} for target in log_targets}
        self.log_destinations = dict.fromkeys(log_destinations, None)
        self.batch_log = {}
        self.batch_log_tf = {}
       
    def layer_summary(self, layer):
        """For Debugging NOT for saving/logging. maybe move to rsc"""
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
        """Initialize lists to log each layers and batches activation."""
        for ix, l in enumerate(self.model.layers[1:]):    
          self.batch_log[f"layer_{ix}"] = []
          self.batch_log_tf[f"layer_{ix}"] = []
          
    def on_batch_end(self, batch, logs=None):
        for ix, l in enumerate(self.model.layers[1:]):
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
         
        #Check Replicas/Distribution 
        print(f"cross-replica-ctxt: {tf.distribute.in_cross_replica_context()}")
        print(f"tf.dist.strat:  {tf.distribute.get_strategy()}") 
        
        #Compare Activation Logging (tf.var vs. keras weight)
        for ix, l in enumerate(self.model.layers[1:]):
            for (batch_k, batch_tf) in zip(self.batch_log[f"layer_{ix}"], self.batch_log_tf[f"layer_{ix}"]): #TODO theres rsc.compare_tensor_lists()
                #print(f"Keras logged activation: {batch_k}")
                #print(f"TF logged activation: {batch_tf}")
                print(f"Compare tensors: {rsc.compare_tensors(batch_k, batch_tf, 'batch_k', 'batch_tf')}")
        
        #raise Exception("Just to break before the rest gets printed.")        
         
        #Compare Results (Aggregators, sat_layer states, both algorithms) 
        for ix, l in enumerate(self.model.layers[1:]):
            if ix==2:
                return
            
            #Two-Pass CovMat
            tp_cov_mat, tp_cov_mat_e = SatFunctions.two_pass_cov(self.batch_log[f"layer_{ix}"])
            #print(f"Two Pass CovMat {l}: {tp_cov_mat}")
            #rsc.compare_tensors(tp_cov_mat, tp_cov_mat_e, "tp_cov_batched", "tp_cov_epoch")
            
            
            #Naive CovMat
            o_s, r_s, s_s, _ = l.sat_layer.get_states()
            naive_cov_mat = SatFunctions.get_cov_mat(o_s, r_s, s_s)
            naive_cov_mat_N = SatFunctions.get_cov_mat(50, r_s, s_s)
            
            rsc.compare_tensors(tp_cov_mat, naive_cov_mat_N)
            
            print(f"o_s: {o_s}")
            
            #Compare Cov-Algorithms
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
            
        