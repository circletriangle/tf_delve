import tensorflow as tf 
from tensorflow import keras 
import numpy as np
import SatLayer
import importlib

importlib.reload(SatLayer)


class mydense(keras.layers.Dense):
    
    def print_args(self, *args, **kwargs):
        for arg in args:    
            print("arg: {}".format(arg))
        for key in kwargs.keys():
            print("kwarg {} -> {}".format(key, kwargs[key]))  
    
    def process_params(self, params):
        if "input_shape_build" in params.keys():
            super().build(params["input_shape_build"])
        if "init_weights" in params.keys():
            self.set_weights(params["init_weights"])
        if "output_shape" in params.keys():
            output_shape = params["output_shape"]
        
        if not self.weights==[]:
            self.sat_layer = SatLayer.sat_layer(input_shape=output_shape, name="sat_l_"+str(self.name))    

    def __init__(self, custom_params=None, *args, **kwargs):
        super(mydense, self).__init__(*args, **kwargs) 
        if custom_params:
            self.process_params(custom_params)
        else:
            raise NameError("Extra Parameters not found!")    
          
            
                    
    def call(self, inputs):
        out = super().call(inputs)
        #self.add_metric(tf.reduce_sum(out), name="my_placeholder_metric_"+self.name, aggregation='mean')
        o, r, s, = self.sat_layer(out)
        self.add_metric(o, name="observed_samples_sat_"+str(self.name), aggregation="mean")
        #self.add_metric(self.sat_layer.result(), name="result_"+str(self.name) , aggregation="mean")
        #print("out shp {} l: {}".format(self.name, out.get_shape()) )
        return out
        
    @classmethod 
    def from_config_params(cls, params, config):
        new = cls(custom_params=params, **config)
        return new
        


def clone_fn(old_layer):
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
    


def satify_model(model, compile_args=[]):
    assert model.output_shape, "Output Shape not defined! (Call model to build)"
    
    #TODO build/import callback for satmod modification
    
    clone = tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=clone_fn)
    
    z_in = np.ones(shape=(20,4))
    assert np.allclose(model.predict(z_in), clone.predict(z_in)), "Wrong: predictions don't match original"
    
    if not compile_args==[]:
        clone.compile(**compile_args)
    
    
    return clone 
    

    
    
    