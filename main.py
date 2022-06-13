import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import random
import pandas as pd
import random
import datetime
import pytz

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def unzip(filename):
  unzip = zipfile.ZipFile(filename)
  unzip.extractall()
  unzip.close()

def plot_loss_curves(history):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"]))

  plt.figure()
  plt.plot(epochs,loss, label="training_loss")
  plt.plot(epochs,val_loss, label="validation_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()
  plt.figure()
  plt.plot(epochs,accuracy, label="training_accuracy")
  plt.plot(epochs,val_accuracy, label="validation_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()

def pred_and_plot(filename, model, class_names, img_shape=224):
  img = tf.io.read_file(filename)
  image = tf.image.decode_image(img)
  img = tf.image.resize(image, size=(img_shape,img_shape))
  img = img/255.
  img = tf.expand_dims(img,axis=0)
  if len(class_names) == 2: 
    predict = model.predict(img)
    predict = class_names[int(tf.round(predict))]
  else: 
    predict = model.predict(img)
    predict = class_names[int(predict.argmax())]
  plt.imshow(image)
  plt.title(predict)
  plt.axis(False)
  
def len_files(filename):  
  for dirpath, dirnames, filenames in os.walk(filename):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")

def view_random_image(dir):
  folder = random.choice(os.listdir(dir))
  files=random.choice(os.listdir(dir+"/"+folder))
  img = mpimg.imread(dir+"/"+folder+"/"+files)
  plt.imshow(img)
  plt.title(folder)
  plt.axis(False)
  plt.show()
  print(f"Image file Name: {files}")

def tiny_cnn(train_data,test_data,epochs,loss,output,metrics="accuracy",input_size=(244,244,3),seed=0):
  tf.random.set_seed(seed)

  model = Sequential([
    Conv2D(10,3,activation="relu",input_shape=input_size),
    MaxPool2D(),
    Conv2D(10,3,activation="relu"),
    MaxPool2D(),
    Conv2D(10,3,activation="relu"),
    Conv2D(10,3,activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(output, activation="softmax")
  ])

  model.compile(loss=loss,
                  optimizer=Adam(),
                  metrics=[metrics])

def save_and_load(model,test_data,filename="saved_model"):
  model.save(filename)
  loaded_model = tf.keras.models.load_model(filename)
  print("loaded model")
  print(loaded_model.evaluate(test_data))
  print("unloaded model")
  print(model.evaluate(test_data))
  return loaded_model
  history = model.fit(train_data,
                      epochs=epochs,
                      steps_per_epoch=len(train_data),
                      validation_data=test_data,
                      validation_steps=len(test_data))
  
def create_tensorborad_callback(dir_name, experiment_name,india=False):
  if india == False:
    tz = 'Asia/Riyadh'
  else:
    tz = 'Asia/Kolkata'
  log_dir = dir_name +"/" + experiment_name+"/"+datetime.datetime.now(pytz.timezone(tz)).strftime("%y%m%d-%I%M%p")
  tensorborad_callback = TensorBoard(log_dir=log_dir)
  print(f"Saving to:{log_dir}")
  return tensorborad_callback

import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import random
import pandas as pd
import random
import datetime
import pytz

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def unzip(filename):
  unzip = zipfile.ZipFile(filename)
  unzip.extractall()
  unzip.close()

def plot_loss_curves(history):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"]))

  plt.figure()
  plt.plot(epochs,loss, label="training_loss")
  plt.plot(epochs,val_loss, label="validation_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()
  plt.figure()
  plt.plot(epochs,accuracy, label="training_accuracy")
  plt.plot(epochs,val_accuracy, label="validation_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()

def pred_and_plot(filename, model, class_names, img_shape=224):
  img = tf.io.read_file(filename)
  image = tf.image.decode_image(img)
  img = tf.image.resize(image, size=(img_shape,img_shape))
  img = img/255.
  img = tf.expand_dims(img,axis=0)
  if len(class_names) == 2: 
    predict = model.predict(img)
    predict = class_names[int(tf.round(predict))]
  else: 
    predict = model.predict(img)
    predict = class_names[int(predict.argmax())]
  plt.imshow(image)
  plt.title(predict)
  plt.axis(False)
  
def len_files(filename):  
  for dirpath, dirnames, filenames in os.walk(filename):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")

def view_random_image(dir):
  folder = random.choice(os.listdir(dir))
  files=random.choice(os.listdir(dir+"/"+folder))
  img = mpimg.imread(dir+"/"+folder+"/"+files)
  plt.imshow(img)
  plt.title(folder)
  plt.axis(False)
  plt.show()
  print(f"Image file Name: {files}")

def tiny_cnn(train_data,test_data,epochs,loss,output,metrics="accuracy",input_size=(244,244,3),seed=0):
  tf.random.set_seed(seed)

  model = Sequential([
    Conv2D(10,3,activation="relu",input_shape=input_size),
    MaxPool2D(),
    Conv2D(10,3,activation="relu"),
    MaxPool2D(),
    Conv2D(10,3,activation="relu"),
    Conv2D(10,3,activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(output, activation="softmax")
  ])

  model.compile(loss=loss,
                  optimizer=Adam(),
                  metrics=[metrics])

def save_and_load(model,test_data,filename="saved_model"):
  model.save(filename)
  loaded_model = tf.keras.models.load_model(filename)
  print("loaded model")
  print(loaded_model.evaluate(test_data))
  print("unloaded model")
  print(model.evaluate(test_data))
  return loaded_model
  history = model.fit(train_data,
                      epochs=epochs,
                      steps_per_epoch=len(train_data),
                      validation_data=test_data,
                      validation_steps=len(test_data))
  
def create_tensorborad_callback(dir_name, experiment_name,india=False):
  if india == False:
    tz = 'Asia/Riyadh'
  else:
    tz = 'Asia/Kolkata'
  log_dir = dir_name +"/" + experiment_name+"/"+datetime.datetime.now(pytz.timezone(tz)).strftime("%y%m%d-%I%M%p")
  tensorborad_callback = TensorBoard(log_dir=log_dir)
  print(f"Saving to:{log_dir}")
  return tensorborad_callback

def model_from_url(model_url,num_classes,img_size):
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False,
                                           name="feature_extraction",
                                           input_size = (img_size,img_size,3))
  
  if num_classes == 2:
    activation = "sigmoid"
  else:
    activation = "softmax"
  model = Sequential([
                      feature_extractor_layer,
                      Dense(num_classes,activation=activation,name="output_layer")
  ])
  return model
