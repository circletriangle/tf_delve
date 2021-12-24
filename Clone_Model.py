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
        
        TODO Tidy up and remove tracking and auxiliary attributes from here. 
        (Compartmentalize all that to sublayer, makes it easier to copy to conv too)
        -> plus small tweaks like consistent (custom_)params naming, dropping *args from __init__() signature,...
        
        TODO !Expand the params dict and pass delta, cov_alg, ... also through it!!
    """    
    
    @classmethod 
    def from_config_params(cls, params, config):
        """
            Extends from_config() functionality by passing extra arguments/parameters.

            CALLS: mydense.__init__()
            CALLED BY: clone_fn()
            
            PARAMETERS: 
                params (dict): information on previously built original (e.g. existing weights)
                config (dict?): base config info that only get's passed to keras.layers.Dense.from_config()
            
            RETURNS:
                new (mydense layer): the created layer passed back to the overall cloning process    
        """    
        
        new = cls(custom_params=params, **config) #Should I even ** the config here? Try changing it later
        return new
        
    def print_args(self, *args, **kwargs):
        """
            Auxiliary/Debugging function to print/inspect args/kwargs.
            
            Keep for DEBUGGING and then it's SAFE TO DELETE LATER
            / OR MOVE TO RSC
            
            PARAMETERS:
                args (list/tuple): list of arguments
                kwargs (dict): dict of keyword arguments
        """
        
        for arg in args:    
            print("arg: {}".format(arg))
        for key in kwargs.keys():
            print("kwarg {} -> {}".format(key, kwargs[key]))  
    
    def process_params(self, params):
        """
            Set layer weights from params (in order) before 
            adding Fields that change the config-signature.
            
            CALLS:          super().build(), super().set_weights(), sat_layer.__init__(), log_layer.__init__()
            CALLED BY:      my_dense/self.__init__()
            
            1. build() needs input shape. 
            2. init_weights can be set in built layer.
            3. Add new fields (set_weights() can't take weights with old signature now)
                3.1 Create saturation-sublayer with output_shape (track states, get saturation).
                3.2 Create aggregators (track states) (metrics not added to layer)
                
            PARAMETERS:
                params (dict): contains original layer's state/weights for copying    
                
            #TODO add cov_alg, delta to arguments passed to sublayers ~ <- Unecessary if we go with expose-only from now on.
            #TODO is it good to user super().set_weights()... here? I guess as long as it works for conv and dense versions.
        """
    
        #COPY ORIGINAL STATE
        if "input_shape_build" in params.keys():
            super().build(params["input_shape_build"])
        if "init_weights" in params.keys():
            self.set_weights(params["init_weights"])
        if "output_shape" in params.keys():
            output_shape = params["output_shape"]
        
        #ADD SUBLAYERS
        if not self.weights==[]:
            #self.sat_layer = SatLayer.sat_layer(input_shape=output_shape, name="sat_l_"+str(self.name))
            self.log_layer = SatLayer.log_layer(input_shape=output_shape, dtype=self.dtype, name="log_l_"+str(self.name))    
        
        #SAVE INFO ON LAYER PROPERTIES (#TODO DEPRECATED? DELETE? STATES FULLY TRACKED IN SUBLAYER?)
        features = output_shape[1]
        self.states = ['o_s', 'r_s', 's_s']
        shapes = [(),(features),(features,features)]
        self.features = tf.dtypes.cast(features, dtype=tf.float64)

    def __init__(self, custom_params=None, *args, **kwargs):
        """
            Initializes a Dense Object and then extends it with process_params()
            
            CALLS:      self.process.params()
            CALLED BY:  from_config_params()
            
            PARAMETERS:
                custom_params (dict): parameters contain info on the previously built original (e.g. existing weights)
                args, kwargs: eventual additional arguments to pass to super().__init__()
        """

        #Initialize basic Dense Object
        super(mydense, self).__init__(*args, **kwargs) #TODO super points to mydense not keras.Dense -> check if that leaves something uninitialized
        
        #
        if custom_params:
            self.process_params(custom_params)
        else:
            raise NameError("Extra Parameters not found!")          
                            
    def call(self, inputs):
        """
            Passes base-class/Dense activation to sat_layer/aggregators before returning it.
            
            CALLS:          Dense/super().call(),   self.satlayer.__call__()  (not satlayer.call() i think)
            CALLED BY:      model forward pass
            
            #TODO add control_depencies to ensure sat_layer gets updates
            
            PARAMETERS:
                inputs (keras tensor): activation of the previous/input layer
            RETURNS: 
                out (keras tensor): own/dense activation    
        
        """    
        out = super().call(inputs)
        
        #_o, _r, _s, = self.sat_layer(out)
        #o, r, s, = self.sat_layer.get_update_values(out)
        
        _ = self.log_layer(out)
         
        #self.add_metric(o, aggregation=tf.VariableAggregation.SUM, name="tf_aggregation")
        
        #with tf.control_depencies([_o, _r, _s]):
        return out

class myconv(keras.layers.Conv2D):
    """
        Extends keras.conv2d by adding a sublayer and saturation functionalities during cloning. (like mydense)
        
        Copy-Pasted from mydense so some details may still have to be adapted.
        
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
                (3.2 Create aggregators (track states) (metrics not added to layer))
        """
    
        if "input_shape_build" in params.keys():
            super().build(params["input_shape_build"])
        if "init_weights" in params.keys():
            self.set_weights(params["init_weights"])
        if "output_shape" in params.keys():
            output_shape = params["output_shape"]
        
        #ADD SUBLAYERS
        if not self.weights==[]:
            #self.sat_layer = SatLayer.sat_layer(input_shape=output_shape, name="sat_l_"+str(self.name))
            self.log_layer = SatLayer.log_layer(input_shape=output_shape, dtype=self.dtype, name="log_l_"+str(self.name))    
        
        #MAYBE DEPRECATED TO TRACK ANYTHING HERE AND NOT EVEN IN THE SUBLAYER.
        features = output_shape[1]
        self.states = ['o_s', 'r_s', 's_s']
        shapes = [(),(features),(features,features)]
        self.features = tf.dtypes.cast(features, dtype=tf.float64)

    def __init__(self, custom_params=None, *args, **kwargs):
        """
            Init Conv2D Object and extend it with process_params()
            
            PARAMETERS:
                custom_params (dict): information to get get basic part of custom layer in the same state as the original.
        """
        super(myconv, self).__init__(*args, **kwargs) #TODO super points to mydense not keras.Dense -> check if that leaves something uninitialized
        if custom_params:
            self.process_params(custom_params)
        else:
            raise NameError("Extra Parameters not found!")          
                            
    def call(self, inputs):
        """
            Pass activation to sat_layer/(aggregators) and return it.
            
            PARAMETERS:
                inputs (tf.Tensor): input from last layer in model.
                
            RETURNS:
                out (tf.Tensor): output of the layer.    
            
            TODO Decide whether to flatten here (-> cleaner here, then log_layer doesn't have to care what class's sublayer it is)
            
            TODO add control_depencies to ensure sat_layer gets updates? apparently eager and autograph both don't need it, but ...
        """    
        out = super().call(inputs)
        
        #_o, _r, _s, = self.sat_layer(out)
        #o, r, s, = self.sat_layer.get_update_values(out)
        
        #_ = self.log_layer(out)
        #_ = self.log_layer.update(out) 
         
        #self.add_metric(o, aggregation=tf.VariableAggregation.SUM, name="tf_aggregation")
        
        #with tf.control_depencies([_o, _r, _s]):
        return out

        
                  
def clone_fn(old_layer):
    """
        Clones and extends a single layer if it's class is saturation capable.
    
        This function is passed to clone_model() to specify how to copy each original layer. 
        
        - Default case / Layer class not supported (-> same as when no clone_fn passed): 
            creates new instance of base-class with '@cls from_config(original_config)'. 

        - Custom case / Layer class supported:
            creates instance of class extending base-class with '@cls from_config_params(params, original_config)'
            passing additional argument params containing layer-information needed for custom intialization.

        CALLS: mydense.from_config_params() /(old_layer.__class__.from_config())
        CALLED BY: clone_model(original_model)
        
        PARAMETERS:
            old_layer (keras Layer): single layer to be transcribed to new model
            
        RETURNS:
            new_layer (keras/custom Layer): layer instance to be used by the new model      
        
    """
    
    # DEBUGGING / INSPECTION
    if False:
        print(old_layer.__class__)
        print(old_layer.__dict__) #-> super insightful       
        new_layer = old_layer.__class__.from_config(old_layer.get_config())    
        
    # CLONE DENSE LAYER - ADDED FSS FUNCTIONALITY:
    if old_layer.__class__==tf.keras.layers.Dense:
        config = old_layer.get_config()
        assert old_layer.output_shape, "layer {} output shape undefined! (never called)".format(old_layer.name) 
        params = {"input_shape_build": old_layer.input_shape,
                  "init_weights": old_layer.get_weights(),
                  "output_shape": old_layer.output_shape} #for never called models out_shp not def -> add dry run in func?
        new_layer = mydense.from_config_params(params, config)
        return new_layer
    
    # CLONE CONV LAYER - ADDED FSS FUNCTIONALITY:
    if old_layer.__class__==tf.keras.layers.Conv2D:
        config = old_layer.get_config()
        assert old_layer.output_shape, "layer {} output shape undefined! (never called)".format(old_layer.name) 
        params = {"input_shape_build": old_layer.input_shape,
                  "init_weights": old_layer.get_weights(),
                  "output_shape": old_layer.output_shape} #for never called models out_shp not def -> add dry run in func?
        new_layer = myconv.from_config_params(params, config)
        return new_layer
    
    
    #TODO Uncomment the Copying of old layer weights in the unsupported layer block: !!
    # CLONE UNSUPPORTED LAYER NORMALLY:    #TODO -> or find and call the default clone_fn() here~!!
    if True:
        #try and see if i need to copy weights separately, think not but can't google rn
        new_layer = old_layer.__class__.from_config(old_layer.get_config())
        #new_layer.call(np.ones(shape=(1,)+old_layer.input_shape[1:] ))
        #new_layer.call(np.ones(shape=old_layer.input_shape ))
        #new_layer.set_weights(old_layer.get_weights())
        return new_layer
        
        print(old_layer.__class__.__name__)
        return old_layer.__class__.from_config(old_layer.get_config())
    
    
    
    
    
def satify_model(model, compile_dict=None, batch_size=None):
    """
        Clones a keras model to add FSS functionality 
        and manages details of the process / cloned model.
        
        CALLS:      clone_model() (-> clone_fn())
        CALLED BY:  User
        
        PARAMETERS:
            model (keras.model):    original compiled model as blueprint/base
            compile_dict (dict):    compile-arguments of original model
            batch_size (int ?):     batch_size that the clone will take and be able to process
            
        RETURNS:
            clone (keras model):    model capable of tracking and generating values relating to FSS   
    """
    
    
    
    
    assert model.output_shape, "Output Shape not defined! (Call model to build)"
    
    print(f"Model output_shape: {model.output_shape}")
    
    clone = tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=clone_fn)
    
    #TODO check if clone and model prediction diff comes from non-mydense layers not having weights copied.
    for clone_layer, og_layer in zip(clone.layers, model.layers): 
        print(f"Layer {clone_layer.__class__.__name__}")  
        #print(f"config, weights: {clone_layer.get_config()}, {clone_layer.get_weights()}")
        #if clone_layer.__class__ == og_layer.__class__:
        #    assert not clone_layer.weights==[], "Clone Layer weights uninitialized"
        #    assert (np.allclose(clone_layer.get_weights(), og_layer.get_weights()) \
        #        or clone_layer.__class__ == mydense), f"Layer weights don't match! Class: {og_layer.__class__.__name}" 
    
    
    """
    example_input_shp = (1,) + model.input_shape[1:]
    z_in = np.ones(shape=example_input_shp)
    """
    
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
    
    if compile_dict==None:
        compile_dict={}
    
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
    

    
    
    