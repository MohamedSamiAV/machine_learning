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
import time
import tensorflow_datasets as tfds

from tensorflow.keras import mixed_precision
from google import colab
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense,Input, GlobalAveragePooling2D, Activation
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory


def unzip(filename, walk=True):
  """
  Unzips filename into the current working directory.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()
  if walk:
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
  return img
  
def create_tensorborad_callback(experiment_name,dir_name="tensorboard",tz='Asia/Riyadh'):
  log_dir = dir_name +"/" + experiment_name+"/"+datetime.datetime.now(pytz.timezone(tz)).strftime("%y%m%d-%I%M%p")
  tensorborad_callback = TensorBoard(log_dir=log_dir)
  print(f"Saving to:{log_dir}")
  return tensorborad_callback


def checkpoint_callback(filename,save_best_only=False,save_weights_only=False,monitor='val_loss',verbose=1):
  checkpoint_path = "checkpoint/"+filename+".ckpt"
  checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  monitor=monitor,
                                                  verbose=verbose,
                                                  save_best_only=save_best_only,
                                                  save_weights_only=save_weights_only)
  return checkpoint

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

def compare_historys(history_1, history_2=None, initial_epochs=5,figsize=(10,10)):
  
  # Get original history measurements
  acc = history_1.history["accuracy"]
  loss = history_1.history["loss"]
  
  epochs = range(len(history_1.history['loss']))

  val_acc = history_1.history["val_accuracy"]
  val_loss = history_1.history["val_loss"]

  # Combine original history with new history
  if history_2 != None:
    total_acc = acc + history_2.history["accuracy"]
    total_loss = loss + history_2.history["loss"]

    total_val_acc = val_acc + history_2.history["val_accuracy"]
    total_val_loss = val_loss + history_2.history["val_loss"]

  if history_2 != None:
    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, acc, label='training_accuracy')
    plt.plot(epochs, val_acc, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

  if history_2 != None:
    # Make plots
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs, initial_epochs],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs, initial_epochs],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def save_and_load(model,test_data,evaluate=False,filename="saved_model"):
  filenames = "/drive/MyDrive/models/saved_model"+filename
  model.save(filenames)
  loaded_model = tf.keras.models.load_model(filenames)
  if evaluate==True:
    loaded_eva = loaded_model.evaluate(test_data)
    model_eva = model.evaluate(test_data)
    print("loaded model")
    print(loaded_eva)
    print("unloaded model")
    print(model_eva)
    print(np.isclose(np.array(model_eva),np.array(loaded_eva)))
  return loaded_model

def make_confusion_matrix(y_true,y_pred,classes=None,figsize=(15,15), fontsize=20, savefig=""): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(15, 15)).
    text_size: Size of output figure text (default=15).
    savefig: Provide filename to save confusion matrix to file (default="").
  
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
  ax.xaxis.set_ticklabels(labels,rotation=70)
  ax.yaxis.set_ticklabels(labels,rotation=0)

  ## Display the visualization of the Confusion Matrix.
  plt.show()

  # Save the figure to the current working directory
  if savefig != "":
    fig.savefig(savefig)

def f1_score(y_true,y_pred,class_names,figsize=(12,25)):
  class_f1_scores = {}
  report = classification_report(y_true,y_pred,output_dict=True)
  for k,v in report.items():
    if k == "accuracy":
      break
    else:
      class_f1_scores[class_names[int(k)]] = v["f1-score"]
  f1_scores = pd.DataFrame({"class_names":list(class_f1_scores.keys()),
                            "f1-score":list(class_f1_scores.values())}).sort_values("f1-score",ascending=False)
  fig, ax = plt.subplots(figsize)
  scores = ax.barh(range(len(f1_scores)),f1_scores["f1-score"].values)
  ax.set_yticks(range(len(f1_scores)))
  ax.set_yticklabels(f1_scores["class_names"])
  ax.set_xlabel("F1-score")
  ax.set_title("F1-score for each class names")
  ax.invert_yaxis()

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

def wrong_100(test_dir,y_true,y_prob,class_names,start_index,img_shape=224):
  y_preds = y_prob.argmax(axis=1)
  filepaths = []
  images_to_view = 9
  for filepath in test_data.list_files(test_dir+"/*/*.jpg",shuffle=False):
    filepaths.append(filepath.numpy())
  pred_df = pd.DataFrame({"img_path":filepaths,
                          "y_true":y_true,
                          "y_pred":y_preds,
                          "pred_conf":y_prob.max(axis=1),
                          "y_true_classname":[class_names[i] for i in y_true],
                          "y_pred_classname":[class_names[i] for i in y_preds]})
  pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
  top_100_wrong = pred_df[pred_df["pred_correct"]==False].sort_values("pred_conf",ascending=False)[:100]
  plt.figure(figsize=(15,12))
  for i, row in enumerate(top_100_wrong[start_index:start_index+images_to_view].itertuples()):
    plt.subplot(3,3,i+1)
    img = tf.io.read_file(row[1])
    image = tf.image.decode_image(img,3)
    img = tf.image.resize(image, size=(img_shape,img_shape))
    img = img/255.
    _, _, _,_,pred_conf, y_true_classname, y_pred_classname,_ = row
    plt.imshow(img)
    plt.title(f"actual:{y_true_classname}, pred: {y_pred_classname}\n {pred_conf*100}.2:f")
    plt.axis(False)
  return top_100_wrong

def pred_and_plot(filename, model, class_names,rescale=True, img_shape=224):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  img = tf.io.read_file(filename)
  image = tf.image.decode_image(img,3)
  img = tf.image.resize(image, size=(img_shape,img_shape))
  if rescale:
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
  
  return predict
