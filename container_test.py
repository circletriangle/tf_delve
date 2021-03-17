import tensorflow as tf 
from tensorflow import keras as keras
from tensorflow.python.keras import backend as K
#import os
import numpy as np
#import datetime
#import importlib

import SatCallbacks
import Clone_Model
import rsc


#      TRAIN MODEL EAGER ENABLED MNIST CLASSIFICATION
###########################################

batch_size = 20

train, test = rsc.get_mnist()


model = rsc.get_model_unitlist(hidden_layers_spec=[128])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



history = model.fit(train[0],
                    train[1],
                    batch_size=batch_size,
                    epochs=5, 
                    steps_per_epoch=None, 
                    validation_data=test, 
                    validation_steps=1, 
                    callbacks=[])



#      CLONE MODEL EAGER ENABLED MNIST CLASSIFICATION
###########################################

clone = Clone_Model.satify_model(model)

new_cb = SatCallbacks.sat_logger()

sat_cb = SatCallbacks.sat_results()

history = clone.fit(train[0],
                    train[1],
                    batch_size=10,
                    epochs = 2,
                    steps_per_epoch=5,
                    validation_data=test, 
                    validation_steps=2, 
                    callbacks=[new_cb])
              