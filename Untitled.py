#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install tensorflow==2.12.0


# In[1]:


pip install pandas


# In[3]:


pip install tensorflow


# In[4]:


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# In[7]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix


# In[8]:


import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)


# In[10]:


my_data_dir = './dataset/cell_images'
os.listdir(my_data_dir)


# In[11]:


test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'


# In[12]:


os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[200]


# In[13]:


para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[200])


# In[45]:


plt.imshow(para_img)
print("soundariyan \n212222230146\n")


# In[15]:


dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


# In[16]:


sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)


# In[17]:


image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )


# In[18]:


image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)


# In[19]:


model = models.Sequential()


# In[20]:


# Add convolutional layers
model.add(layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


# In[21]:


model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


# In[22]:


model.add(layers.Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


# In[23]:


# Flatten the layer
model.add(layers.Flatten())


# In[24]:


# Add a dense layer
model.add(layers.Dense(128, activation='relu'))


# In[25]:


# Output layer
model.add(layers.Dense(1, activation='sigmoid'))


# In[26]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[27]:


model.summary()


# In[28]:


batch_size = 16
help(image_gen.flow_from_directory)
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')


# In[29]:


train_image_gen.batch_size
len(train_image_gen.classes)
train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)


# In[30]:


train_image_gen.class_indices


# In[31]:


results = model.fit(train_image_gen,epochs=3,
                              validation_data=test_image_gen
                             )


# In[32]:


losses = pd.DataFrame(model.history.history)
print("SOUNDARIYAN MN\n212222230146\n")
losses[['loss','val_loss']].plot()


# In[33]:


model.metrics_names


# In[34]:


print("Soundariyan MN\n212222230146\n")
model.evaluate(test_image_gen)


# In[35]:


pred_probabilities = model.predict(test_image_gen)


# In[36]:


print("Soundariyan MN\n212222230146\n")
test_image_gen.classes


# In[37]:


predictions = pred_probabilities > 0.5
print("Soundariyan MN\n212222230146\n")
print(classification_report(test_image_gen.classes,predictions))


# In[38]:


print("Soundariyan MN\n212222230146\n")
confusion_matrix(test_image_gen.classes,predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




