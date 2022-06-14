import zipfile
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import numpy as np
import random
import pandas as pd
import random
import datetime
import pytz
import itertools

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense,Input, GlobalAveragePooling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory


def unzip(filename):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

def pred_and_plot(filename, model, class_names, img_shape=224):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
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
  plt.axis(False);
  
def dir_walk(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def view_random_image(dir):
  folder = random.choice(os.listdir(dir))
  files=random.choice(os.listdir(dir+"/"+folder))
  img = mpimg.imread(dir+"/"+folder+"/"+files)
  plt.imshow(img)
  plt.title(folder)
  plt.axis(False)
  plt.show()
  print(f"Image file Name: {files}")

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
  
def create_tensorborad_callback(dir_name, experiment_name,tz='Asia/Riyadh'):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    tz: time-zonefor datetime, default is Saudi Arabia
  """
  log_dir = dir_name +"/" + experiment_name+"/"+datetime.datetime.now(pytz.timezone(tz)).strftime("%y%m%d-%I%M%p")
  tensorborad_callback = TensorBoard(log_dir=log_dir)
  print(f"Saving to:{log_dir}")
  return tensorborad_callback

def model_from_url(model_url,num_classes,img_size):
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False,
                                           name="feature_extraction",
                                           input_shape = (img_size,img_size,3))
  
  if num_classes == 2:
    activation = "sigmoid"
  else:
    activation = "softmax"
  model = Sequential([
                      feature_extractor_layer,
                      Dense(num_classes,activation=activation,name="output_layer")
  ])
  return model

def make_confusion_matrix(y_true,y_pred,classes=None,figsize=(15,15), fontsize=20, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(15, 15)).
    text_size: Size of output figure text (default=15).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """ 
  cm = confusion_matrix(y_true,y_pred)
  
  fig, ax = plt.subplots(figsize=figsize)
  ax = sns.heatmap(cm, annot=True, cmap='Reds', fmt='g')
  ax.set_title('Seaborn Confusion Matrix with labels\n\n');
  ax.set_xlabel('\nPredicted Values')
  ax.set_ylabel('Actual Values ');
  ## Ticket labels - List must be in alphabetical order
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)

  ## Display the visualization of the Confusion Matrix.
  plt.show()

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

# Function to evaluate: accuracy, precision, recall, f1-score

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results
