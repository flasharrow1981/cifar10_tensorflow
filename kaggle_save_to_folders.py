
# coding: utf-8

# In[2]:


from __future__ import print_function
import traceback
import sys
import glob
import os
import csv
import shutil

INPUT = '/home/dp/down/cifar10/cifar-10-unpack/train' #训练图片
OUTPUT1= '/home/dp/down/cifar10/cifar-10-unpack/classify/train' #目标文件夹train
OUTPUT2= '/home/dp/down/cifar10/cifar-10-unpack/classify/test' #目标文件夹test
CSV='/home/dp/down/cifar10/cifar-10-unpack/trainLabels.csv' #训练标签


def classify_cifar10():
    csv_reader = csv.DictReader(open(CSV, encoding='utf-8'))
    tag_dict= dict()
    for row in csv_reader:
        tag_dict[row['id']]=row['label']
    #print(tag_dict)

    list=os.listdir(INPUT)
    #训练集
    for i in range(0,int(len(list)*0.9)):
        path=os.path.join(INPUT,list[i])
        filename=list[i].split(".")[0]
        
        if os.path.isfile(path):
            tag=tag_dict[filename]
            
            out_path=os.path.join(OUTPUT1,tag)
            #print(os.path.join(out_path,list[i]))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            shutil.copy(path,os.path.join(out_path,list[i]))
            #print(path)
       #测试集
    for i in range(int(len(list)*0.9),len(list)):
        path=os.path.join(INPUT,list[i])
        filename=list[i].split(".")[0]
        
        if os.path.isfile(path):
            tag=tag_dict[filename]
            
            out_path=os.path.join(OUTPUT2,tag)
            #print(os.path.join(out_path,list[i]))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            shutil.copy(path,os.path.join(out_path,list[i]))
            #print(path)      
                    
        
        
        
    
classify_cifar10()
       

