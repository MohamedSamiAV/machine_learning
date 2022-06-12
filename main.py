import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import random

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
  predict = model.predict(img)
  predict = class_names[int(tf.round(predict))]
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
