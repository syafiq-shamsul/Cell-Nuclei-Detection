# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:19:35 2022

@author: syafiq
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from scipy import io

#1 Data Loading
#1.1 Prepare a list for image and mask
train_images=[]
train_masks=[]
test_images=[]
test_masks=[]
train_directory = r"C:\Users\syafi\Documents\AI Class\Deep Learning\Datasets\data-science-bowl-2018-2\train"
test_directory = r"C:\Users\syafi\Documents\AI Class\Deep Learning\Datasets\data-science-bowl-2018-2\test"

#%%
#1.2 Load the images
train_image_dir = os.path.join(train_directory,'inputs')
for image_file in os.listdir(train_image_dir):
    img = cv2.imread(os.path.join(train_image_dir,image_file))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    train_images.append(img)
    
#1.3 Load the masks
train_mask_dir = os.path.join(train_directory,'masks')
for mask_file in os.listdir(train_mask_dir):
    mask = cv2.imread(os.path.join(train_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    train_masks.append(mask)
    
test_image_dir = os.path.join(test_directory,'inputs')
for image_file in os.listdir(test_image_dir):
    img = cv2.imread(os.path.join(test_image_dir,image_file))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    test_images.append(img)
    
#1.3 Load the masks
test_mask_dir = os.path.join(test_directory,'masks')
for mask_file in os.listdir(test_mask_dir):
    mask = cv2.imread(os.path.join(test_mask_dir,mask_file),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    test_masks.append(mask)
    
#%%
#1.4 Convert images and masks into numpy array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%%
#1.5 Check some examples
plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = train_images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot = train_masks[i]
    plt.imshow(mask_plot,cmap='gray')
    plt.axis('off')
plt.show()

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    img_plot = test_images[i]
    plt.imshow(img_plot)
    plt.axis('off')
plt.show()

plt.figure(figsize=(10,4))
for i in range(1,4):
    plt.subplot(1,3,i)
    mask_plot = test_masks[i]
    plt.imshow(mask_plot,cmap='gray')
    plt.axis('off')
plt.show()

#%%
# Data preprocessing
#2.1 Expand the mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np,axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np,axis=-1)
# Check the mask output
print(train_masks[0].min(),train_masks[0].max())
print(test_masks[0].min(),test_masks[0].max())

#%%
#2.2 Change the mask value (1. normalize the value, 2. encode into numerical encoding)
train_converted_masks = np.ceil(train_masks_np_exp/255)
train_converted_masks = 1 - train_converted_masks

test_converted_masks = np.ceil(test_masks_np_exp/255)
test_converted_masks = 1 - test_converted_masks

#%%
#2.3 Normalize the images
train_converted_images = train_images_np / 255.0

test_converted_images = test_images_np / 255.0

#%%

#2.4 Do train-test split
from sklearn.model_selection import train_test_split
SEED = 12345
x_train,x_val,y_train,y_val= train_test_split(train_converted_images,train_converted_masks,test_size=0.2,random_state=SEED)

#%%
# Convert the numpy array data into the tensor slice
train_x = tf.data.Dataset.from_tensor_slices(x_train)
val_x = tf.data.Dataset.from_tensor_slices(x_val)
train_y = tf.data.Dataset.from_tensor_slices(y_train)
val_y = tf.data.Dataset.from_tensor_slices(y_val)
test_x = tf.data.Dataset.from_tensor_slices(test_converted_images)
test_y = tf.data.Dataset.from_tensor_slices(test_converted_masks)

#%%
#2.6 Zip tensor slice into dataset
train = tf.data.Dataset.zip((train_x,train_y))
val = tf.data.Dataset.zip((val_x,val_y))
test = tf.data.Dataset.zip((test_x,test_y))

#%%
#2.7 Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = 800//BATCH_SIZE
VALIDATION_STEPS = 200//BATCH_SIZE
train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train = train.prefetch(buffer_size=AUTOTUNE)
val = val.batch(BATCH_SIZE).repeat()
val = val.prefetch(buffer_size=AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

#%%

#3 Prepare model
# We are going to create a modified version of U-net
base_model = tf.keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

# Use the activations of those layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

# Define the upsampling stack
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

# Function to create the entire modified U-net
def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

  # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

    x = last(x)
  
    return tf.keras.Model(inputs=inputs, outputs=x)

# Define the model
OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)

#%%
# Compile the model and display the model structure
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
#%%
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
for images, masks in train.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image,sample_mask])
    
#%%
#Create a function to process predicted mask
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

# Create a function to display prediction
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)[0]])
    else:
        display([sample_image, sample_mask,create_mask(model.predict(sample_image[tf.newaxis, ...]))[0]])

# Custom callback to display result during training
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
    
# Tensorboard callback
import datetime
log_dir = r"C:\Users\syafi\Documents\AI Class\Deep Learning\Tensorboard\logs_" + datetime.datetime.now().strftime("%d%m%Y - %H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=1,profile_batch=0)

# Start to do training
EPOCH = 20

history = model.fit(train,epochs=EPOCH,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,validation_data=val,callbacks=[DisplayCallback(),tb_callback])
#%%

#Test evaluation
test_loss, test_accuracy = model.evaluate(test)
print(f"Test loss = {test_loss}")
print(f"Test accuracy = {test_accuracy}")

#%%
# Deploy model by using the show_prediction function created before
show_predictions(test,3)





