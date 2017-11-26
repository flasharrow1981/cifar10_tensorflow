
# coding: utf-8

# In[11]:


# %load read_cifar10_direct


# In[5]:


# %load read_cifar10_direct.py


# In[6]:


# %load read_cifar10_direct.py


# In[30]:


from __future__ import print_function
import traceback
import sys
import glob
import os
import tensorflow as tf
#import matplotlib.image as mpimg # mpimg 用于读取图片
from PIL import Image
import numpy as np

directory= '/home/dp/down/cifar10/cifar-10-unpack/classify' #目标文件夹
classlist=list()

def read_cifar10():
         
    images, labels = [], []
    # 读取训练集
    train_dir=os.path.join(directory,'train')
    train_classlist=os.listdir(train_dir)

    filenames=list()
    labels=list()
    for classes in train_classlist:
        class_path=os.path.join(train_dir,classes)
        filelist=os.listdir(class_path)
        print('classname='+classes)
        for file in filelist:
            filefullName=os.path.join(class_path,file)
            #image_string = tf.read_file(filefullName)
            #image_decoded = tf.image.decode_png(image_string, channels=3)
            #image_resized = tf.image.resize_images(image_decoded, [32, 32])
            #image_decoded = mpimg.imread(filefullName) 
            image_decoded = Image.open(filefullName)
            image_decoded = np.array(image_decoded, dtype=np.uint8)
            
            # 此时已经是一个 np.array 了，可以对它进行任意处理
            image_decoded.reshape(32, 32, 3)
            image_decoded = image_decoded.astype(float)
            images.append(image_decoded)
            labels.append(train_classlist.index(classes))
            #print(image_decoded)
        break   

    #training_data = np.hstack(images, labels)
    #np.random.shuffle(training_data)
    #images = training_data[:, :-1]
    #labels = training_data[:, -1]
    print(len(images))
    print(len(labels))
    training_data = np.hstack((images, labels))
    images = np.array(images, dtype='float')
    labels = np.array(labels, dtype='int')
    
    
    print(labels[1:20])
    #print(images,labels)
    #self.train_images, self.train_labels = images, labels
    
# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
   
    #one_hot = tf.one_hot(label, 10) #one_hot = tf.one_hot(label, NUM_CLASSES)
    return image_resized, label
    #return image_resized, one_hot
read_cifar10()




