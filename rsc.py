import tensorflow as tf
import numpy as np
from tensorflow import keras as keras
from tensorflow.keras import layers
import importlib
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input # to be used on the data for vgg16
from tensorflow.keras.utils import to_categorical

###############################################################################
#  DATA
###############################################################################


def get_cifar10(one_hot=True, in_dtype='float64'):
    """
        Returns tuples of training/testing cifar10 (data, labels).
    
        RETURNS: 
            training data (tuple): cifar10 data and one-hot labels, pixel-values scaled to [0;1], dtype float64
            testing data (tuple): cifar10 data and one-hot labels, pixel-values scaled to [0;1], dtype float64
    """
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # one hot encode target values
    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    
    # Cast to float64 and normalize values to [0,1] 255?
    x_train, x_test = x_train.astype(in_dtype) / 255.0, x_test.astype(in_dtype) / 255.0

    print(f"Shape of cifar10 y_train: {y_train.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def download_data():
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"      
           
    train_file_path = tf.keras.utils.get_file("titanic_train.csv", TRAIN_DATA_URL, cache_dir="./data")
    test_file_path = tf.keras.utils.get_file("titanic_eval.csv", TEST_DATA_URL, cache_dir="./data")

    return train_file_path, test_file_path

def get_dataset(file_path, batch_size=20, num_epochs=1, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size, # Artificially small to make examples easier to show.
        na_value="?",
        num_epochs=num_epochs,
        ignore_errors=False, 
        **kwargs)

    return dataset

def show_batch(dataset):
    np.set_printoptions(precision=3, suppress=True)
    for feature_batch, label_batch in dataset.take(1):
        print("'survived': {}".format(label_batch))
        print("features:")
        for key, value in feature_batch.items():
            #print("  {!r:20s}: {}".format(key, value))
            print("{:20s}: {}".format(key,value.numpy()))

def show_dataset_tensor(dataset, num_batches=-1):
    for ix, batch in enumerate(dataset):
        print(f"Batch number {ix}:")
        print(batch)
        if ix == num_batches:
            break

def show_batch_tensor(dataset):
    np.set_printoptions(precision=3, suppress=True)
    for features, labels in dataset.take(1):
        print("'survived': {}".format(labels))
        print("features: {}".format(features.numpy))

def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label

def onehot_label(features, label):
    return features, tf.one_hot(label, on_value=1, off_value=0, dtype=tf.int32, depth=2)

def get_titanic_dataset(path_train='./data/datasets/titanic_train.csv', path_test='./data/datasets/titanic_eval.csv', batch_size=None):
    CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone'] #if theres no column_name row in csv
    SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare'] #for selecting only some columns
    LABEL_COLUMN = 'survived' #what column to use as label for prediction (idk how exactly to pass both, the constructor has only one arg for labels)
    LABELS = [0, 1]
    DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]

    flag_dict = {
        'column_names': CSV_COLUMNS,
        'label_name': LABEL_COLUMN,
        'column_defaults': DEFAULTS,
        'select_columns': SELECT_COLUMNS,
        'prefetch_buffer_size': None
    }
    
    raw_train_data = get_dataset(path_train, **flag_dict)
    raw_test_data = get_dataset(path_test, **flag_dict)

    train_pack = raw_train_data.map(pack).map(onehot_label)
    test_pack = raw_test_data.map(pack).map(onehot_label)

    if batch_size:
        train = train_pack.batch(batch_size)
        test = test_pack.batch(batch_size)
        return train, test
    
    return train_pack, test_pack
        
def get_t_data_broadcast_label(dim=1):
    train_pack, test_pack = get_titanic_dataset()
    def broadcast_label(features, label):
        return features, label * dim
    #map_fn = lambda (features, label) : features, label * dim
    return train_pack.map(broadcast_label), test_pack.map(broadcast_label)

def broadcast_label(data, dim=1):
    assert isinstance(data, tf.python.data.ops.dataset_ops.MapDataset), "Label Broadcast input not of type MapDataset but type: {}".format(type(data))
    def broadcast_label_inner(features, label):
        return features, label * dim
    return data.map(broadcast_label_inner)
  
def get_data(query=None):
    """
    Calls functions to get one-hot encoded datasets as training and test tuples.
    Args: dict query specifying dataset to return and options like resolution, dtype, etc.
    Returns: tuple (x_train, y_train), tuple (x_train, y_train)
    """    
    
    if query==None:
        raise Exception('No Dataset specified!')    
    
    if query["dataset"] == "titanic":
        train, test = get_titanic_dataset()    
    if query["dataset"] == "cifar10":
        train, test = get_cifar10()
    
    return train, test
    
def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)
        
def get_cifar10_old_todelete():
    """
    #https://www.tensorflow.org/datasets/catalog/cifar10 <- tf link not keras 
    Dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories
    Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
    x_train, x_test: uint8 arrays of RGB image data with 
     shape (num_samples, 3, 32, 32) if tf.keras.backend.image_data_format() is 'channels_first', 
     or (num_samples, 32, 32, 3) if the data format is 'channels_last'.
    y_train, y_test: uint8 arrays of category labels (integers in range 0-9) each with shape (num_samples, 1).
    """
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()      
    
    # Cast to float64 and normalize values to [0,1] 255?
    x_train, x_test = x_train.astype('float64') / 256, x_test.astype('float64') / 256   
    
    # in keras example out-layer Dense(10) works with class-vectors without 
    # one-hot/to_categorical encoding? + Casting to float64? TODO
    #y_train, y_test = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='uint8') 
    
    return (x_train, y_train), (x_test, y_test)
        
###############################################################################
#  MODELS
###############################################################################

def get_model(data, type="default", metrics=None, callbacks=None, depth=4, out_feat=2):
    
    if metrics==None:
        metrics=[]
    if callbacks==None:
        callbacks=[]    
    
    #https://www.tensorflow.org/guide/keras/train_and_evaluate/
    #inputs = keras.Input(shape=(784,), name="digits")
    inputs = keras.Input(shape=(4,), name="model_input")
    x = layers.Dense(21, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(22, activation="relu", name="dense_2")(x)
    x = layers.Dense(23, activation="relu", name="dense_3")(x)
    x = layers.Dense(24, activation="relu", name="dense_4")(x)
    outputs = layers.Dense(units=2, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model
    
def get_model_slim():
        
    inputs = keras.Input(shape=(4,), name="model_input")
    x = layers.Dense(3, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(3, activation="relu", name="dense_2")(x)
    x = layers.Dense(2, activation="relu", name="dense_3")(x)
    x = layers.Dense(2, activation="relu", name="dense_4")(x)
    outputs = layers.Dense(units=2, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model    

def get_model_mnist(data=None):
    
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)])
    
    return model

def get_model_unitlist(data=None, hidden_layers_spec=None):
    
    if hidden_layers_spec == None:
        hidden_layers_spec=[128]
    
    #using model.add() smoother
    in_layer = [tf.keras.layers.Flatten(input_shape=(28,28))]
    hidden_layers = [tf.keras.layers.Dense(width, activation='relu') for width in hidden_layers_spec]
    out_layer = [tf.keras.layers.Dense(10)]
    layers = in_layer + hidden_layers + out_layer
    
    model = tf.keras.Sequential(layers)
    
    return model
        
def get_functional_api_autoencoder():
    encoder_input = keras.Input(shape=(28, 28, 1), name="img")
    x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Conv2D(16, 3, activation="relu")(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)

    encoder = keras.Model(encoder_input, encoder_output, name="encoder")
    encoder.summary()

    x = layers.Reshape((4, 4, 1))(encoder_output)
    x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
    decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
    autoencoder.summary()

def get_vgg16(query=None):
    """
        Adds layers on top of the core vgg16 layers.
        
        Parameters: query (dict): specifications for vgg16 options
        Returns: model (keras.model)
    """
    
    if query==None: 
        query={}
    
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(32,32,3))
    
    x = tf.keras.layers.Flatten()(vgg16.output)
    x = tf.keras.layers.Dense(units=512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=10, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=vgg16.inputs, outputs=x)
    
    return model
    
###############################################################################
#  UTILS
###############################################################################

def compare_tensors(t1, t2, name1="tensor 1", name2="tensor 2"):
    
    print(f"\nComparing {name1} and {name2}:\n")
    
    shapes = [tf.shape(t1), tf.shape(t2)]
    print(f"Shape {name1}: {shapes[0]}\nShape {name2}: {shapes[1]}")
    
    equal_elements = tf.math.equal(t1, t2) 
    all_equal = tf.math.reduce_all(equal_elements, axis=None)
    if tf.executing_eagerly() and all_equal.numpy() == True:
        print(f"Tensors {name1} and {name2} are equal! \n")
        return
    
    diff = t1 - t2 #Not abs() 
    diff_rsum = tf.math.reduce_sum(diff, axis=None)
    diff_rsum_avg = diff_rsum / tf.size(t1, out_type=tf.dtypes.float64)
    print(f"Mean Difference of elements: {diff_rsum_avg}")
    
    diff_ratio1 = tf.math.abs(diff / t1)
    diff_ratio2 = tf.math.abs(diff / t2)
    diff_ratio_avg1 = tf.math.reduce_sum(diff_ratio1, axis=None) / tf.size(t1, out_type=tf.dtypes.float64)
    diff_ratio_avg2 = tf.math.reduce_sum(diff_ratio2, axis=None) / tf.size(t2, out_type=tf.dtypes.float64)
    print(f"Avg Ratio Diff/Value of {name1} elements: {diff_ratio_avg1}")
    print(f"Avg Ratio Diff/Value of {name2} elements: {diff_ratio_avg2}")
    print(f"Avg of elementwise Ratios Diff/{name1}, Diff/{name2}: {diff_ratio_avg1}, {diff_ratio_avg2}")
    max_diff_ratio1 = tf.math.reduce_max(diff_ratio1, axis=None)
    max_diff_ratio2 = tf.math.reduce_max(diff_ratio2, axis=None)
    print(f"Max Ratio Diff/Value {name1}: {max_diff_ratio1}")
    print(f"Max Ratio Diff/Value {name2}: {max_diff_ratio2}") #check argmax?
    
    print("\n")
    #TODO add range of diff, isequal, hash, rsum, dtype, head,...
    #TODO plot matrix of diffs, ratios etc. (heatmap eg)

    
def compare_tensor_lists(l1, l2, names1=None, names2=None):
    #Untested
    
    if not (names1 and names2):
        compare_tensors(t1, t2)
        
    if isinstance(names1, list) and isinstance(names2, list):
        for t1, t2, n1, n2 in zip(l1, l2, names1, names2):
            compare_tensors(t1, t2, n1, n2)
            
    if isinstance(names1, str) and isinstance(names2, str):
        for ix, (t1, t2) in enumerate(zip(l1, l2)):
            compare_tensors(t1, t2, names1+f"_{ix}", names2+f"_{ix}")   
            

def layer_summary(layer):
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
        compare_tensors(val_dict[s][0], val_dict[s][1], "Layer_"+s, "Aggregator_"+s)
