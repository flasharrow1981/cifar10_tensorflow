
# coding: utf-8

# In[2]:


# %load cifar10.py
import pickle
import numpy
import random
import matplotlib.pyplot as plt
import platform
import cv2
import os
import tensorflow as tf
from PIL import Image
import numpy as np


class Corpus:
    
    def __init__(self):
        #self.load_cifar10('data/CIFAR10_data')
        self.load_cifar10_dataset('/home/dp/down/cifar10/cifar-10-unpack/classify') #目标文件夹')
        self._split_train_valid(valid_rate=0.9)
        self.n_train = self.train_images.shape[0]
        self.n_valid = self.valid_images.shape[0]
        self.n_test = self.test_images.shape[0]
        
    def _split_train_valid(self, valid_rate=0.9):
        images, labels = self.train_images, self.train_labels 
        thresh = int(images.shape[0] * valid_rate)
        self.train_images, self.train_labels = images[0:thresh,:,:,:], labels[0:thresh]
        self.valid_images, self.valid_labels = images[thresh:,:,:,:], labels[thresh:]
    #从分类文件夹中读取数据    
    def load_cifar10_dataset(self, directory):
        
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
                #filenames.append(filefullName)
                #image_string = tf.read_file(filefullName)
                #image_decoded = tf.image.decode_png(image_string, channels=3)
                #image_resized = tf.image.resize_images(image_decoded, [32, 32])
                image_decoded = Image.open(filefullName)
                image_decoded = np.array(image_decoded, dtype=np.uint8)

                # 此时已经是一个 np.array 了，可以对它进行任意处理
                image_decoded.reshape(32, 32, 3)
                image_decoded = image_decoded.astype(float)
                images.append(image_decoded)
                #labels.append(classes)
                labels.append(train_classlist.index(classes))
                #print(filefullName)       
        #print(filenames,labels)   
        
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        
        training_data = np.hstack(images, labels)
        np.random.shuffle(training_data)
        images = training_data[:, :-1]
        labels = training_data[:, -1]
        self.train_images, self.train_labels = images, labels
        
        
        # 读取测试集
        images, labels = [], []
        test_dir=os.path.join(directory,'test')
        test_classlist=os.listdir(test_dir)
    
        filenames=list()
        labels=list()
        for classes in test_classlist:
            class_path=os.path.join(test_dir,classes)
            filelist=os.listdir(class_path)
            print('classname='+classes)
            for file in filelist:
                filefullName=os.path.join(class_path,file)
                #filenames.append(filefullName)
                #image_string = tf.read_file(filefullName)
                #image_decoded = tf.image.decode_png(image_string, channels=3)
                #image_resized = tf.image.resize_images(image_decoded, [32, 32])
                image_decoded = Image.open(filefullName)
                image_decoded = np.array(image_decoded, dtype=np.uint8)

                # 此时已经是一个 np.array 了，可以对它进行任意处理
                image_decoded.reshape(32, 32, 3)
                image_decoded = image_decoded.astype(float)
                images.append(image_decoded)
                
                #labels.append(classes)
                labels.append(train_classlist.index(classes))
                #print(filefullName)       
        #print(filenames,labels)   
        
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        test_data = np.hstack(images, labels)
        np.random.shuffle(test_data)
        images = test_data[:, :-1]
        labels = test_data[:, -1]
        self.test_images, self.test_labels = images, labels
        
    
        
        
    
    # 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [32, 32])

        #one_hot = tf.one_hot(label, 10) #one_hot = tf.one_hot(label, NUM_CLASSES)
        return image_resized, label
        #return image_resized, one_hot
    
    def load_cifar10(self, directory):
        # 读取训练集
        images, labels = [], []
        for filename in ['%s/data_batch_%d' % (directory, j) for j in range(1, 2)]:#in range(1, 6)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
            for i in range(len(cifar10[b"labels"])):
                image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                image = numpy.transpose(image, (1, 2, 0))
                image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        self.train_images, self.train_labels = images, labels
        # 读取测试集
        images, labels = [], []
        for filename in ['%s/test_batch' % (directory)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
            for i in range(len(cifar10[b"labels"])):
                image = numpy.reshape(cifar10[b"data"][i], (3, 32, 32))
                image = numpy.transpose(image, (1, 2, 0))
                image = image.astype(float)
                images.append(image)
            labels += cifar10[b"labels"]
        images = numpy.array(images, dtype='float')
        labels = numpy.array(labels, dtype='int')
        self.test_images, self.test_labels = images, labels
        
    def data_augmentation(self, images, mode='train', flip=False, 
                          crop=False, crop_shape=(24,24,3), whiten=False, 
                          noise=False, noise_mean=0, noise_std=0.01):
        # 图像切割
        if crop:
            if mode == 'train':
                images = self._image_crop(images, shape=crop_shape)
            elif mode == 'test':
                images = self._image_crop_test(images, shape=crop_shape)
        # 图像翻转
        if flip:
            images = self._image_flip(images)
        # 图像白化
        if whiten:
            images = self._image_whitening(images)
        # 图像噪声
        if noise:
            images = self._image_noise(images, mean=noise_mean, std=noise_std)
            
        return images
    
    def _image_crop(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = numpy.random.randint(old_image.shape[0] - shape[0] + 1)
            top = numpy.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left+shape[0], top: top+shape[1], :]
            new_images.append(new_image)
        
        return numpy.array(new_images)
    
    def _image_crop_test(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            new_image = old_image[left: left+shape[0], top: top+shape[1], :]
            new_images.append(new_image)
        
        return numpy.array(new_images)
    
    def _image_flip(self, images):
        # 图像翻转
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            if numpy.random.random() < 0.5:
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
            images[i,:,:,:] = new_image
        
        return images
    
    def _image_whitening(self, images):
        # 图像白化
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
            images[i,:,:,:] = new_image
        
        return images
    
    def _image_noise(self, images, mean=0, std=0.01):
        # 图像噪声
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = old_image
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        new_image[i, j, k] += random.gauss(mean, std)
            images[i,:,:,:] = new_image
        
        return images

