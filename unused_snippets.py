
"""
Two approaches for a class meant to track/aggregate a state that is
continously updated by tensors and accessible to keras as a metric.
Added as metric from callback on_train_begin() and reset on_epoch_end().
I'm not pursuing that because the aggregation (averaging over replicas i think) keras 
does, is not customizable (no single values possible, inprecise)
The same thing is now done by tracking the states as keras-weights
of satlayer that are updated with assign(_add).
"""
class Aggremetric(tf.keras.metrics.Metric):
    """Try version of aggregator that is used by add_metric in call() and tracked."""
    
    def __init__(self, input, name=None, dtype=tf.float64):
        super(Aggremetric, self).__init__(name=name, dtype=dtype)
        
        self.shape = K.int_shape(input)
        self.init = tf.keras.initializers.Zeros()
        self.value = self.add_weight("aggremetric_state",
                                     shape=self.shape,
                                     dtype=tf.float64,
                                     initializer=self.init) 
                
    def update_state(self, y_true, y_pred, sample_weights=None):
        return self.value.assign_add(y_true)
    
    def update_state(self, update_tensor):
        return self.value.assign_add(update_tensor)
    
    def reset_state(self):
        return self.value.assign(self.init(self.shape))
    
    def result(self):
        return self.value    
                              
class Aggregator(tf.keras.metrics.Metric):
    """Aggregate by updating a state by a tensor. """
    
    def __init__(self, shape, name=None, dtype=tf.float64):
        """Track/update one state."""
        super(Aggregator, self).__init__(name=name, dtype=dtype)
        
        #self.init = tf.keras.initializers.Zeros()
        #self.shape = shape
        self.state = self.add_weight("aggr_state_"+name, 
                                    shape=shape,  
                                    dtype=dtype,
                                    initializer=tf.keras.initializers.Zeros())
        self.init_value = tf.zeros(shape=shape, dtype=dtype, name="aggr_init_"+name)
        
    def update_state(self, update_tensor):
        #tf.print(f"AGGREG UPDATESTATE() CALLED: {update_tensor}")
        return self.state.assign_add(update_tensor)
    
    def result(self):
        return self.state
    
    
    def reset_state(self):
        return self.state.assign(self.init_value)
        #K.set_value(self.state, self.init_value)
       
       
#from mydense.call()       
self.aggregators['o_s'].update_state(o) 
self.aggregators['r_s'].update_state(r) 
self.aggregators['s_s'].update_state(s)               
#for name, aggr in self.ag_metrics_tpl:
    #self.model.metrics_names.append(self.name+name)
    #self.model.metrics_tensors.append(aggr)   
#_ = K.print_tensor(s, message="K.print_tensor() in call(/)")
# add_metric() API won't let custom metric aggregate (Why is it not recognized as metrics.Metric subclass?)
#aggr_try = Aggremetric(r, name="Agrmtrc")
#self.add_metric(aggr_try, name="ajsdfklas")
#-> try adding normal aggregators to metrics()
# keras refuses to not mean aggr here
#self.add_metric(o, aggregation=tf.VariableAggregation.SUM, name="tf_aggregation")
        