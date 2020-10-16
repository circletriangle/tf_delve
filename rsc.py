import tensorflow as tf
import numpy as np
from tensorflow import keras as keras
from tensorflow.keras import layers
import importlib

###############################################################################
#  DATA
###############################################################################

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
  
def get_data():
    training_data, test_data = get_titanic_dataset()    
    
    
###############################################################################
#  MODELS
###############################################################################

def get_model(data, type="default", metrics=[], callbacks=[], depth=4, out_feat=2):
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

    