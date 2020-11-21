import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import SatLayer
import importlib
import tensorflow.keras.backend as K
import SatFunctions
import rsc

importlib.reload(SatLayer)
importlib.reload(SatFunctions)
importlib.reload(rsc)

class mydense(keras.layers.Dense):
    """
    For cloning an existing layer (Dense) object and adding 
    a sublayer that computes the saturation metric.
    
    Additional parameters need to be passed to init() through from_config_params(). Those are:
    -Original weights (need to be set in init before adding sublayer; after is impossible because the signature doesn't match)
    -Input shape (needed for build() so the weights can be set)
    -Output shape (needed as input shape of the saturation sublayer)
    
    call() then passes each forward passes activation to the sat-sublayer before returning it.
    """    
    
    @classmethod 
    def from_config_params(cls, params, config):
        """Extends from_config() functionality by passing extra arguments/parameters."""    
        new = cls(custom_params=params, **config)
        return new
        
    def print_args(self, *args, **kwargs):
        for arg in args:    
            print("arg: {}".format(arg))
        for key in kwargs.keys():
            print("kwarg {} -> {}".format(key, kwargs[key]))  
    
    def process_params(self, params):
        """
        Set layer weights from params (in order) before 
        adding Fields that change the config-signature.
        
        1. build() needs input shape. 
        2. init_weights can be set in built layer.
        3. Add new fields (set_weights() can't take weights with old signature now)
            3.1 Create saturation-sublayer with output_shape (track states, get saturation).
            3.2 Create aggregators (track states) (metrics not added to layer)
        """
    
        if "input_shape_build" in params.keys():
            super().build(params["input_shape_build"])
        if "init_weights" in params.keys():
            self.set_weights(params["init_weights"])
        if "output_shape" in params.keys():
            output_shape = params["output_shape"]
        
        if not self.weights==[]:
            self.sat_layer = SatLayer.sat_layer(input_shape=output_shape, name="sat_l_"+str(self.name))    
        
        features = output_shape[1]
        self.states = ['o_s', 'r_s', 's_s']
        shapes = [(),(features),(features,features)]
        self.features = tf.dtypes.cast(features, dtype=tf.float64)

        #InfoA mit names :)
        self.aggregators = {name : Aggregator(shape, name=name+str(self.name)) 
                            for name, shape in zip(self.states,shapes)}
        self.ag_metrics = [self.aggregators[state] for state in self.states]
        self.ag_metrics_tpl = [(state, self.aggregators[state]) for state in self.states]
        
    def __init__(self, custom_params=None, *args, **kwargs):
        """Init Dense Object and extend it with process_params()"""
        super(mydense, self).__init__(*args, **kwargs) 
        if custom_params:
            self.process_params(custom_params)
        else:
            raise NameError("Extra Parameters not found!")    
            
    @property
    def sat_states(self):
        """Should be part of logs that are passed to batch callbacks?
        Without being processed by keras.""" 
        return self.ag_metrics       
            
    @property
    def metrics(self):
        """Trying to get our not added() metrics into metrics-list.
        Scalar metric gets into metrics-list but not history.h.keys() ~kinda hacky
        TODO keras tries to set the non scalar matrices to 0 no broadcasting."""
        return [self.ag_metrics[0]]        
                            
    def call(self, inputs):
        """Pass activation to sat_layer/aggregators and return it.
        TODO add control_depencies to ensure sat_layer gets updates"""    
        out = super().call(inputs)
        
        
        _o, _r, _s, = self.sat_layer(out)
        o, r, s, = self.sat_layer.get_update_values(out)
        
        self.aggregators['o_s'].update_state(o) 
        self.aggregators['r_s'].update_state(r) 
        self.aggregators['s_s'].update_state(s)               
        
        #tf.print(f"ACT TENSOR DENSE.CALL: {out}")
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
        
        #with tf.control_depencies([_o, _r, _s]):
        return out
        

class sat_results(keras.callbacks.Callback):
    
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
        
    #def on_train_begin(self, logs=None):
        #for layer in self.model.layers[1:]:
            #layer.add_metric(Aggremetric())
            #layer.add_metric(layer.aggregators['r_s'], name="r_s_added_callb")    
        
    def on_batch_end(self, batch, logs=None):
        print(f"\nLogs from batch callback: {logs}")
        print(f"\nK.GETVALUE o_s satlayer: {K.get_value(self.model.layers[1].sat_layer.o_s)}" )
        
    def on_epoch_end(self, epoch, logs=None):
        for l in self.model.layers[1:]:
            
            print(f"\nEXECUTING-EAGERLY: {tf.executing_eagerly()}")
            print(f"\nLogs from epoch callback: {logs}")
            _ = K.print_tensor(l.sat_layer.r_s, message="\nK.print_tensor()\n")
            print(f"\nK.GETVALUE o_s satlayer: {K.get_value(l.sat_layer.o_s)}" )
            #print(f"\nK.GETVALUE o_s aggregator: {K.get_value(l.aggregators['o_s'].state)}")
            #print(f"\nK EVAL satlayer result(): {K.eval(l.sat_layer.result())}" )
            
            self.layer_summary(l)
            
            l.sat_layer.reset()
            for s in l.states:
                l.aggregators[s].reset_state()
         


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
    
    
def clone_fn(old_layer):
    """
    This function is to be passed to clone_model() and applied to each original layer,
    defining its cloned version. Usually a layer of the same class is created by the classmethod
    from_config() using config info of the original layer. (If no custom cone_fn() is specified)
    Here instead a layer (mydense) extending the base class is instantiated by from_config_params()
    in order to pass additional arguments on to init that are not covered by from_config().
    
    """
    #print(old_layer.__class__)
    #print(old_layer.__dict__) -> super insightful       
    #new_layer = old_layer.__class__.from_config(old_layer.get_config())    
    if old_layer.__class__==tf.keras.layers.Dense:
        config = old_layer.get_config()
        assert old_layer.output_shape, "layer {} output shape undefined! (never called)".format(old_layer.name) 
        params = {"input_shape_build": old_layer.input_shape,
                  "init_weights": old_layer.get_weights(),
                  "output_shape": old_layer.output_shape} #for never called models out_shp not def -> add dry run in func?
        new_layer = mydense.from_config_params(params, config)
        return new_layer
    else:
        print(old_layer.__class__)
        return old_layer.__class__.from_config(old_layer.get_config())
    
def satify_model(model, compile_dict={}):
    
    assert model.output_shape, "Output Shape not defined! (Call model to build)"
    
    clone = tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=clone_fn)
    
    #z_in = np.ones(shape=(20,4)) TODO get defined input shape
    #z_in = np.ones(shape=model.layers[0].input_shape[-2:])
    z_in = np.ones(shape=(1,28,28))
    assert np.allclose(model.predict(z_in), clone.predict(z_in)), "Cloned Model Predictions don't match Original!"
    
    
    default_compile_dict = {
        'optimizer': model.optimizer.__class__.__name__,
        'loss': model.loss,
        'metrics' : model.metrics,
        'run_eagerly' : True
    }    
    #second dict overwrites conflicting default keys
    merged_dict = {**default_compile_dict, **compile_dict}    
        
    clone.compile(**merged_dict)
    
    
    print("Clone_Model.satify_model() end.")
    return clone 
    

    
    
    