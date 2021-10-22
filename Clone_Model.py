import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import SatLayer
import importlib
import tensorflow.keras.backend as K
import SatFunctions
import rsc
import SatCallbacks

importlib.reload(SatLayer)
importlib.reload(SatFunctions)
importlib.reload(rsc)
importlib.reload(SatCallbacks)

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
        
        #ADD SUBLAYERS
        if not self.weights==[]:
            self.sat_layer = SatLayer.sat_layer(input_shape=output_shape, name="sat_l_"+str(self.name))
            self.log_layer = SatLayer.log_layer(input_shape=output_shape, dtype=self.dtype, name="log_l_"+str(self.name))    
        
        features = output_shape[1]
        self.states = ['o_s', 'r_s', 's_s']
        shapes = [(),(features),(features,features)]
        self.features = tf.dtypes.cast(features, dtype=tf.float64)

    def __init__(self, custom_params=None, *args, **kwargs):
        """Init Dense Object and extend it with process_params()"""
        super(mydense, self).__init__(*args, **kwargs) 
        if custom_params:
            self.process_params(custom_params)
        else:
            raise NameError("Extra Parameters not found!")          
                            
    def call(self, inputs):
        """Pass activation to sat_layer/aggregators and return it.
        TODO add control_depencies to ensure sat_layer gets updates"""    
        out = super().call(inputs)
        
        _o, _r, _s, = self.sat_layer(out)
        o, r, s, = self.sat_layer.get_update_values(out)
        
        _ = self.log_layer(out)
         
        #self.add_metric(o, aggregation=tf.VariableAggregation.SUM, name="tf_aggregation")
        
        #with tf.control_depencies([_o, _r, _s]):
        return out
                  
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
        
        #try and see if i need to copy weights separately, think not but can't google rn
        new_layer = old_layer.__class__.from_config(old_layer.get_config())
        #new_layer.call(np.ones(shape=(1,)+old_layer.input_shape[1:] ))
        #new_layer.call(np.ones(shape=old_layer.input_shape ))
        #new_layer.set_weights(old_layer.get_weights())
        return new_layer
        
        print(old_layer.__class__.__name__)
        return old_layer.__class__.from_config(old_layer.get_config())
    
    
    
    
    
def satify_model(model, compile_dict={}, batch_size=None):
    
    assert model.output_shape, "Output Shape not defined! (Call model to build)"
    
    print(f"Model output_shape: {model.output_shape}")
    
    clone = tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=clone_fn)
    
    # check if clone and model prediction diff comes from non-mydense layers not having weights copied.
    for clone_layer, og_layer in zip(clone.layers, model.layers): 
        print(f"Layer {clone_layer.__class__.__name__}")  
        #print(f"config, weights: {clone_layer.get_config()}, {clone_layer.get_weights()}")
        #if clone_layer.__class__ == og_layer.__class__:
        #    assert not clone_layer.weights==[], "Clone Layer weights uninitialized"
        #    assert (np.allclose(clone_layer.get_weights(), og_layer.get_weights()) \
        #        or clone_layer.__class__ == mydense), f"Layer weights don't match! Class: {og_layer.__class__.__name}" 
    
    example_input_shp = (1,) + model.input_shape[1:]
    z_in = np.ones(shape=example_input_shp)
    
    """
    clone.predict(z_in)
    for c_l, og_l in zip(clone.layers, model.layers):
        if c_l.__class__ == og_l.__class__: 
            c_l.set_weights(og_l.get_weights())
            print(f"Set weights for {c_l}")
    """    
    """
    assert np.allclose(model.predict(z_in), clone.predict(z_in)), \
        f"Cloned Model Predictions don't match Original! \n Original: {model.predict(z_in)}, \n Clone: {clone.predict(z_in)}  \n"
    """
    
    default_compile_dict = {
        'optimizer': model.optimizer.__class__.__name__,
        'loss': model.loss,
        'metrics' : model.metrics,
        'run_eagerly' : True
    }    
    #second dict overwrites conflicting default keys
    merged_dict = {**default_compile_dict, **compile_dict}    
        
    clone.compile(**merged_dict)
 
    return clone 
    

    
    
    