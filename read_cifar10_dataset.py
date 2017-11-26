
# coding: utf-8

# In[2]:


from __future__ import print_function
import traceback
import sys
import glob
import os
import tensorflow as tf

directory= '/home/dp/down/cifar10/cifar-10-unpack/classify' #目标文件夹
classlist=list()

def read_cifar10():
    train_dir=os.path.join(directory,'train')
    classlist=os.listdir(train_dir)
    
    filenames=list()
    labels=list()
    for classes in classlist:
        class_path=os.path.join(train_dir,classes)
        filelist=os.listdir(class_path)
        print('classname='+classes)
        for file in filelist:
            filefullName=os.path.join(class_path,file)
            filenames.append(filefullName)
            labels.append(classes)
            #labels.append(classlist.index(classes))
            #print(filefullName)       
    #print(filenames,labels)    
    tf_filenames=tf.constant(filenames)
    tf_labels=tf.constant(labels)
    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))

    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)

    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    #dataset = dataset.shuffle(buffer_size=50000).batch(32).repeat(10)
    
    iterator = dataset.make_one_shot_iterator()
    #batch_features, batch_labels = iterator.get_next()
    # create TensorFlow Iterator object
    #iterator = Iterator.from_structure(tr_data.output_types,
    #                               tr_data.output_shapes)
    next_element = iterator.get_next()
    
    # create two initialization ops to switch between the datasets
    training_init_op = iterator.make_initializer(dataset)
    

    with tf.Session() as sess:

        # initialize the iterator on the training data
        sess.run(training_init_op)

        # get each element of the training dataset until the end is reached
        #while True:
        try:
            elem = sess.run(next_element)
            print(elem)
            print(elem[0].shape)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
           #break
    
# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [32, 32])
   
    #one_hot = tf.one_hot(label, 10) #one_hot = tf.one_hot(label, NUM_CLASSES)
    return image_resized, label
    #return image_resized, one_hot
read_cifar10()

