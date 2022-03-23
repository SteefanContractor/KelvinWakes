# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Surface wake detection models - Hyperparameter search
#
# The models below take the training images used by Jack Buckley and classify them into Kelvin wake/No Kelvin wake classes. 
# We perform a hyperparameter search over the number of layers, filters per convolutional block, kernel size for convolutions 
# and number of units in the final dense layer and log all trained models on mlflow.


# +
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://ploomber.readthedocs.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# + tags=["parameters"]
# extract_upstream=True in your pipeline.yaml file, if this task has
# dependencies, list them them here (e.g. upstream = ['some_task']), otherwise
# leave as None. Once you modify the variable, reload it for Ploomber to inject
# the cell (On JupyterLab: File -> Reload File from Disk)
upstream = None

# extract_product=False in your pipeline.yaml file, leave this as None, the
# value in the YAML spec will be injected in a cell below. If you don't see it,
# check the Jupyter logs
product = None
num_filters = None
kernel_size = None
dense_units = None
params_names = None
track = None
mlflow_tracking_uri = None
# -


model_params = {k: globals()[k] for k in params_names}
print(model_params)

# your code here...
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Rescaling
import matplotlib.pyplot as plt
from sklearn_evaluation import plot
import atexit
from unittest.mock import Mock
import mlflow
from mlflow.exceptions import MlflowException

if track:
    print('tracking with mlflow...')
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    @atexit.register
    def end_run():
        mlflow.end_run()
else:
    print('tracking skipped...')
    mlflow = Mock()


# +
def train_val_test_split(ds, ds_size, train_test_split=0.9, train_val_split=0.8, val_shuffle=False):
    assert train_test_split <= 1
    assert train_val_split < 1
    
    train_size = int(train_test_split * ds_size)
    
    train_val_ds = ds.take(train_size)    
    test_ds = ds.skip(train_size)
    
    train_size = int(train_val_split * train_size)
    if val_shuffle:
        ds = train_val_ds.shuffle()
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)
    else:
        train_ds = train_val_ds.take(train_size)
        val_ds = train_val_ds.skip(train_size)
    
    return train_ds, val_ds, test_ds

def display(imgtensor, lab=None):
    plt.figure(figsize=(10, 10))
    # plt.subplot(1, numimgs, i+1)
    CLASS = ['Kelvin', 'No Kelvin']
    if lab is not None:
        plt.title(f"label: {CLASS[lab]}")
    plt.imshow(tf.keras.preprocessing.image.array_to_img(imgtensor), cmap='gray')
    plt.axis('off')
    plt.show()

def conv_block(x, filters, kernel_size, activation, padding='same', convstrides=(1,1)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding, strides=convstrides)(x)
    x = MaxPool2D()(x)
    # x = Dropout(dropout)(x)
    return x

def build_model(input_shape, num_filters, kernel_size, dense_units, activation='relu', summary=True):
    num_blocks=len(num_filters)
    inputs = tf.keras.Input(shape=input_shape)
    x = Rescaling(1/255.)(inputs)
    for i in range(num_blocks):
        x = conv_block(x, 
                       filters=num_filters[i],
                       kernel_size=kernel_size,
                       activation=activation
                      )
    x = Flatten()(x)
    x = Dense(dense_units, activation=activation)(x)
    # x = Dropout(dropout[1])(x)
    pred = Dense(2)(x) # 2 classes

    model=tf.keras.Model(inputs=inputs, outputs=pred)
    if summary:
        model.summary()
        
    return model


# +
experiment_name = "surfacewakeimgclassification_hypersearch"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

print(f'experiment id: {experiment_id}')
# -

run = mlflow.start_run(experiment_id=experiment_id)

# + tags=["mlflow-run-id"]
print(run.info.run_id)
# -

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

# +
BATCH_SIZE=16
SHAPE=(128,128,1) # (256,256) 
TRAIN_TEST_SPLIT=0.9
TRAIN_VAL_SPLIT=0.8
SEED=202203
LR=0.0005
EPOCHS = 1

# log original model_params created from pipeline.yaml params and then
# create a new ds_params dict with the above params to log
mlflow.log_params(model_params)
ds_params = {"batch_size": BATCH_SIZE,
             "img_shape" : SHAPE,
             "train_test_split": TRAIN_TEST_SPLIT,
             "train_val_split" : TRAIN_VAL_SPLIT,
             "seed": SEED}
mlflow.log_params(ds_params)

# +
imgdata = tf.keras.utils.image_dataset_from_directory("/srv/scratch/kelvinwakes/imgdata/ship_wake_photos8", 
                                                      image_size=SHAPE[:2], 
                                                      color_mode='grayscale',
                                                      batch_size=BATCH_SIZE, 
                                                      seed=SEED,
                                                      shuffle=True #date                                          
                                                     )
DS_SIZE = len(imgdata)
train_ds, val_ds, test_ds = train_val_test_split(imgdata, ds_size=DS_SIZE, 
                                                 train_test_split=TRAIN_TEST_SPLIT, 
                                                 train_val_split=TRAIN_VAL_SPLIT)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
len(train_ds), len(val_ds), len(test_ds)
# -

model = build_model(SHAPE, num_filters, kernel_size, dense_units)

# +
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
             )

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS,
  verbose=2
)
# -

val_pred = np.argmax(model.predict(val_ds), axis=1)
val_labels = np.concatenate([y for x, y in val_ds], axis=0) 
CLASS = ['Kelvin', 'No Kelvin']
fig, ax = plt.subplots()
plot.confusion_matrix(val_labels, val_pred, CLASS, ax=ax)
mlflow.log_figure(fig, "cm.png")
