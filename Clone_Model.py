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
    
    def __init__(self, custom_params=None, *args, **kwargs):
        super(mydense, self).__init__(*args, **kwargs) 
        
        if custom_params: #TODO process params in separate fun
            if "input_shape_build" in custom_params.keys():
                super().build(custom_params["input_shape_build"])
            if "init_weights" in custom_params.keys():
                self.set_weights(custom_params["init_weights"])
            if "output_shape" in custom_params.keys():
                output_shape = custom_params["output_shape"]
                    
        if not self.weights==[]: #TODO get sat functionality from other file
            self.sat_layer = SatLayer.sat_layer(input_shape=output_shape, name="sat_l_"+str(self.name))
            #self.o_s = tf.Variable(0., name="o_s_"+str(self.name) , trainable=False) # <- update by batch_dimension?
            #self.r_s = tf.Variable(np.zeros(shape=output_shape[1:]), name="r_s"+str(self.name), trainable=False ) #<-shp like output reduced over batch_dim?
            
            
                    
    def call(self, inputs):
        out = super().call(inputs)
        self.add_metric(tf.reduce_sum(out), name="my_placeholder_metric_"+self.name, aggregation='mean')
        o, r, s, = self.sat_layer(out)
        self.add_metric(o, name="observed_samples_sat_"+str(self.name), aggregation="mean")
        #self.o_s.assign_add(out.get_shape()[-1])
        #self.r_s.assign_add(tf.reduce_sum(out, axis=0 ) ) 
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
        params = {"input_shape_build": old_layer.input_shape,
                  "init_weights": old_layer.get_weights(),
                  "output_shape": old_layer.output_shape} #for never called models out_shp not def -> add dry run in func?
        new_layer = mydense.from_config_params(params, config)
        return new_layer
    else:
        print(old_layer.__class__)
    


def satify_model(model):
    #TODO check if out_shp defined/model called before
    #TODO build/import callback for satmod modification
    
    clone = tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=clone_fn)
    
    z_in = np.ones(shape=(20,4))
    assert np.allclose(model.predict(z_in), clone.predict(z_in)), "Wrong: predictions don't match og"
    
    #TODO shift Compilation to here
    
    return clone 
    

    
    
    