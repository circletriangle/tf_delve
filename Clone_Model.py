import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import SatLayer
import importlib

importlib.reload(SatLayer)


class mydense(keras.layers.Dense):
    """
    For cloning an existing layer (Dense) object and extending it by 
    a sublayer that computes the saturation metric.
    
    Additional parameters need to be passed to init() through from_config_params(). Those are:
    -Original weights (need to be set in init before adding sublayer; after is impossible because the signature doesn't match)
    -Input shape (needed for build() so the weights can be set)
    -Output shape (needed as input shape of the saturation sublayer)
    
    call() then passes each forward passes activation to the sat-sublayer before returning it.
    """    
    
    def print_args(self, *args, **kwargs):
        for arg in args:    
            print("arg: {}".format(arg))
        for key in kwargs.keys():
            print("kwarg {} -> {}".format(key, kwargs[key]))  
    
    def process_params(self, params):
        """
        Aux. function of init() to process additional layer-information.
        ->First build() needs the input shape.
        ->After layer is built the weights can be set to init weights.
        Only after the weights are copied can variables/fields be added, because after adding to the layer
        set_weights() cannot align the signatures of the layers weights.
        ->Create sublayer with the outputshape as inputshape.
        """
    
        if "input_shape_build" in params.keys():
            super().build(params["input_shape_build"])
        if "init_weights" in params.keys():
            self.set_weights(params["init_weights"])
        if "output_shape" in params.keys():
            output_shape = params["output_shape"]
        
        if not self.weights==[]:
            self.sat_layer = SatLayer.sat_layer(input_shape=output_shape, name="sat_l_"+str(self.name))    

    def __init__(self, custom_params=None, *args, **kwargs):
        """Inits basic dense object and extends it with process_params()"""
        super(mydense, self).__init__(*args, **kwargs) 
        if custom_params:
            self.process_params(custom_params)
        else:
            raise NameError("Extra Parameters not found!")    
          
            
                    
    def call(self, inputs):
        """
        Passes act. super().call() to sat_layer then returns it.
        """    
        out = super().call(inputs)
        #self.add_metric(tf.reduce_sum(out), name="my_placeholder_metric_"+self.name, aggregation='mean')
        o, r, s, = self.sat_layer(out)
        self.add_metric(o, name="observed_samples_sat_"+str(self.name), aggregation="mean")
        #self.add_metric(self.sat_layer.result(), name="result_"+str(self.name) , aggregation="mean")
        #print("out shp {} l: {}".format(self.name, out.get_shape()) )
        return out
        
    @classmethod 
    def from_config_params(cls, params, config):
        """Extends from_config() functionality by passing extra arguments/parameters."""    
        new = cls(custom_params=params, **config)
        return new
        
class sat_results(keras.callbacks.Callback):
     
    def on_epoch_end(self, epoch, logs=None):
        for l in self.model.layers[1:]:
            print("\nLayer {} sat_result: {}".format(l.name,l.sat_layer.result()))
            print("Observed samples: {}".format(l.sat_layer.o_s.numpy()))
            l.sat_layer.reset()
            #print("Observed samples after reset: {}".format(l.sat_layer.o_s.numpy()))
                    


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
    assert np.allclose(model.predict(z_in), clone.predict(z_in)), "Wrong: predictions don't match original"
    
    if not bool(compile_dict):
        compile_dict = {
            'optimizer': model.optimizer.__class__.__name__,
            'loss': model.loss,
            'metrics' : model.metrics,
            'run_eagerly' : tf.executing_eagerly()
        }    
    clone.compile(**compile_dict)
    
    
    print("Clone_Model.satify_model() end.")
    return clone 
    

    
    
    