import dataset
import tensorflow as tf
import time
import sys
import os
import json
from datetime import timedelta
import math
import random
import numpy as np
from layer_function import create_flatten_layer,create_weights,create_biases,create_convolutional_layer,create_fc_layer

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

if len(sys.argv)<2:
    print("python3 train_cnn.py json_file")
    quit

#load model configuration from json file
mdata = json.load(open(sys.argv[1]))
batch_size=int(mdata["train"]["batch_size"])
classes=list(mdata["classes"].keys())
num_classes=len(classes)
validation_size=mdata["train"]["validation_size"]
img_size=mdata["train"]["img_size"]
num_channels=mdata["train"]["channel"]
#make running directory
run_dir=mdata["train"]["run_dir"]
if not os.path.exists(run_dir):
        os.makedirs(run_dir)
#copy training data to the running directory
trn_dat_dir=run_dir+"/"+"training_data"
if not os.path.exists(trn_dat_dir):
        os.makedirs(trn_dat_dir)
for cls in classes:
    if os.path.exists(trn_dat_dir+"/"+cls):
        os.system("rm -rf "+trn_dat_dir+"/"+cls)
    cmd="cp -R "+mdata["classes"][cls]+" "+trn_dat_dir+"/"+cls
    os.system(cmd)

# Load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(trn_dat_dir, img_size, classes, validation_size=validation_size)
print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

# Declare placeholders
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#Create neural networks
layers=mdata["model"].keys()
current_layer=""
ordered_layer=[]
for layer in layers:
    if mdata["model"][layer]["input"]=="x":
        current_layer=layer
        ordered_layer.append(layer)
        break
for i in range(len(layers)-1):
    for layer in layers:
        if mdata["model"][layer]["input"]==current_layer:
            current_layer=layer
            ordered_layer.append(layer)
            break
n=0
for layer in ordered_layer:
    if n==0:
        print("Layer: "+layer)
        if mdata["model"][layer]["type"]=="flatten":
            nn_model=create_flatten_layer(x)
        elif mdata["model"][layer]["type"]=="convolutional":
            nn_model=create_convolutional_layer(input=x,
                            num_input_channels=num_channels,
                            conv_filter_size=mdata["model"][layer]["filter_size"],
                            num_filters=mdata["model"][layer]["num_filters"],
                            activation=mdata["model"][layer]["activation"],
                            pooling=mdata["model"][layer]["pooling"],
                            win_strd_size=mdata["model"][layer]["win_strd_size"])
        else:
            print("The first layer type must be flatten or convolutional")
            quit()
    else:
        shape=nn_model.get_shape().as_list()
        print("Shape: "+str(shape))
        print("Layer: "+layer)
        if mdata["model"][layer]["type"]=="flatten":
            nn_model=create_flatten_layer(nn_model)
        elif mdata["model"][layer]["type"]=="convolutional":
            try:
                activation=mdata["model"][layer]["activation"]
            except KeyError:
                activation="no"
            nn_model=create_convolutional_layer(input=nn_model,
                            num_input_channels=shape[3],
                            conv_filter_size=mdata["model"][layer]["filter_size"],
                            num_filters=mdata["model"][layer]["num_filters"],
                            activation=activation,
                            pooling=mdata["model"][layer]["pooling"],
                            win_strd_size=mdata["model"][layer]["win_strd_size"])
        elif mdata["model"][layer]["type"]=="fc":
            try:
                n_outputs=mdata["model"][layer]["num_outputs"]
            except KeyError:
                n_outputs=num_classes #if no n_outputs is supplied, assume the readout layer
            try:
                dropout=mdata["model"][layer]["dropout"]
            except KeyError:
                dropout="no"
            try:
                activation=mdata["model"][layer]["activation"]
            except KeyError:
                activation="no"
            nn_model=create_fc_layer(input=nn_model,
                            num_inputs=shape[1],
                            num_outputs=n_outputs,
                            dropout=dropout,
                            activation=activation)
        else:
            print("Hidden types must be flatten, fc or convolutional")
            quit()
    n=n+1
shape=nn_model.get_shape().as_list()
print("Shape: "+str(shape))

y_pred=tf.nn.softmax(nn_model,name='y_pred')
y_pred_cls=tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=nn_model,labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = eval("tf.train."+mdata["train"]["optimizer"]+"Optimizer(learning_rate="+str(mdata["train"]["learning_rate"])+").minimize(cost)")
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer()) 


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, run_dir+"/"+mdata["train"]["model_name"]) 


    total_iterations += num_iteration

train(num_iteration=mdata["train"]["num_iterations"])
