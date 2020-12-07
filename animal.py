# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:02:51 2020

@author: Pedro
"""
#Replicating: https://www.kaggle.com/khotijahs1/predict-multiple-animal-faces-use-different-ml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import matplotlib.image as implt
from PIL import Image 
import seaborn as sns
import cv2 as cs2
import os

import warnings
warnings.filterwarnings('ignore')

## import Keras and its module for image processing and model building
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization

#copying the pretrained models to the cache directory
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#copy the Xception models
!cp ../input/keras-pretrained-models/xception* ~/.keras/models/
#show
!ls ~/.keras/models

train_path = "../input/animal-faces/afhq/train"
test_path = "../input/animal-faces/afhq/val"

train_cat = "../input/animal-faces/afhq/train/cat"
train_dog = "../input/animal-faces/afhq/train/dog"
train_wild = "../input/animal-faces/afhq/train/wild"

test_cat = "../input/animal-faces/afhq/val/cat"
test_dog = "../input/animal-faces/afhq/val/dog"
test_wild = "../input/animal-faces/afhq/val/wild"

# VISUALIZATION
category_names = os.listdir(train_path) # output: ['cat', 'dog','wild']
nb_categories = len(category_names) # output: 3
train_images = []

for category in category_names:
    folder = train_path + "/" + category
    train_images.append(len(os.listdir(folder)))

sns.barplot(y=category_names, x=train_images).set_title("Number Of Training Images Per Category");


img = load_img('../input/animal-faces/afhq/train/cat/flickr_cat_000002.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array 
print('image shape: ', x.shape)

print('Train Cat Image')
plt.imshow(img)
plt.show()


img = load_img('../input/animal-faces/afhq/train/dog/flickr_dog_000002.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array 
print('Train Dog Image')
plt.imshow(img)
plt.show()


img = load_img('../input/animal-faces/afhq/train/wild/flickr_wild_000002.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array 
print('Train Wild Image')
plt.imshow(img)
plt.show()

#Processing Dataset
#1. We should first resize all the images
img_size = 50
cat_train = []
dog_train = []
wild_train = []
label = []

for i in os.listdir(train_cat): # all train cat images
    if os.path.isfile(train_path + "/cat/" + i): # check image in file
        cat = Image.open(train_path + "/cat/" + i).convert("L") # converting grey scale 
        cat = cat.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        cat = np.asarray(cat)/255 # bit format
        cat_train.append(cat)
        label.append(0)
        
for i in os.listdir(train_dog): # all train dog images
    if os.path.isfile(train_path + "/dog/" + i): # check image in file
        dog = Image.open(train_path + "/dog/" + i).convert("L") # converting grey scale 
        dog = dog.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        dog = np.asarray(dog)/255 # bit format
        dog_train.append(dog)
        label.append(1)
        
        
for i in os.listdir(train_wild): # all train wild images
    if os.path.isfile(train_path + "/wild/" + i): # check image in file
        wild = Image.open(train_path + "/wild/" + i).convert("L") # converting grey scale 
        wild = wild.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        wild = np.asarray(wild)/255 # bit format
        wild_train.append(wild)
        label.append(2)
        
x_train = np.concatenate((cat_train,dog_train,wild_train),axis=0) # training dataset
x_train_label = np.asarray(label) # label array containing 0, 1, and 2
x_train_label = x_train_label.reshape(x_train_label.shape[0],1)

print("cat:",np.shape(cat_train) , "dog:",np.shape(dog_train), "wild:",np.shape(wild_train))
print("train_dataset:",np.shape(x_train), "train_values:",np.shape(x_train_label))

# Visualizing Training data
print(x_train_label[0])
plt.imshow(cat_train[0])

# Visualizing Training data
print(x_train_label[1])
plt.imshow(dog_train[0])

# Visualizing Training data
print(x_train_label[2])
plt.imshow(wild_train[0])

#Scaling down the train set and test set images:
#cat: (5153, 50, 50) dog: (4739, 50, 50) wild: (4738, 50, 50)
#train_dataset: (14630, 50, 50) train_values: (14630, 1)
#test_dataset: (1500, 50, 50) test_values: (1500, 1)
#label 1 for cat, label 2 for dog and label 3 for wild

img_size = 50
cat_test = []
dog_test = []
wild_test = []
label = []

for i in os.listdir(test_cat): # all test cat images
    if os.path.isfile(test_path + "/cat/" + i): # check image in file
        cat = Image.open(test_path + "/cat/" + i).convert("L") # converting grey scale 
        cat = cat.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        cat = np.asarray(cat)/255 # bit format
        cat_test.append(cat)
        label.append(0)
        
for i in os.listdir(test_dog): # all test dog images
    if os.path.isfile(test_path + "/dog/" + i): # check image in file
        dog = Image.open(test_path + "/dog/" + i).convert("L") # converting grey scale 
        dog = dog.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        dog = np.asarray(dog)/255 # bit format
        dog_test.append(dog)
        label.append(1)
        
        
for i in os.listdir(test_wild): # all test wild images
    if os.path.isfile(test_path + "/wild/" + i): # check image in file
        wild = Image.open(test_path + "/wild/" + i).convert("L") # converting grey scale 
        wild = wild.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        wild = np.asarray(wild)/255 # bit format
        wild_test.append(wild)
        label.append(2)
        
x_test = np.concatenate((cat_test,dog_test,wild_test),axis=0) # training dataset
x_test_label = np.asarray(label) # label array containing 0, 1, and 2
x_test_label = x_test_label.reshape(x_test_label.shape[0],1)

print("cat:",np.shape(cat_test) , "dog:",np.shape(dog_test), "wild:",np.shape(wild_test))
print("test_dataset:",np.shape(x_test), "test_values:",np.shape(x_test_label))

# Visualizing Training data
print(x_test_label[0])
plt.imshow(cat_test[0])

# Visualizing Training data
print(x_test_label[0])
plt.imshow(dog_test[0])

# Visualizing Training data
print(x_test_label[0])
plt.imshow(wild_test[0])

x = np.concatenate((x_train,x_test),axis=0) # counttrain_data

y = np.concatenate((x_train_label,x_test_label),axis=0) # count:  test_data
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) # flatten 3D image array to 2D
print("images:",np.shape(x), "labels:",np.shape(y))

#Next step, we need to determine the amount of data for train and test. 
#You can modify test_size and see how it affects the accuracy. Let's split!

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

print("Train Number: ", number_of_train)
print("Test Number: ", number_of_test)

x = np.concatenate((x_train,x_test),axis=0) # count train_data

y = np.concatenate((x_train_label,x_test_label),axis=0) # count test_data
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) # flatten 3D image array to 2D
print("images:",np.shape(x), "labels:",np.shape(y))

x_train = X_train.T
x_test = X_test.T
y_train = y_train.T
y_test = y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


#1. Logistic Regression Classification

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
test_acc_logregsk = round(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)
train_acc_logregsk = round(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)

# with GridSearchCV
from sklearn.model_selection import GridSearchCV

grid = {
    "C": np.logspace(-4, 4, 20),
    "penalty": ["l1","l2"]
}
lg=LogisticRegression(random_state=42)
log_reg_cv=GridSearchCV(lg,grid,cv=10,n_jobs=-1,verbose=2)
log_reg_cv.fit(x_train.T,y_train.T)
print("accuracy: ", log_reg_cv.best_score_)

models = pd.DataFrame({
    'Model': ['LR with sklearn','LR with GridSearchCV' ],
    'Train Score': [train_acc_logregsk, "-"],
    'Test Score': [test_acc_logregsk, log_reg_cv.best_score_*100]
})
models.sort_values(by='Test Score', ascending=False)

#2. SVM (Support Vector Machine) classification
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# We define the SVM model
svmcla = OneVsRestClassifier(BaggingClassifier(SVC(C=5,kernel='rbf',random_state=42, probability=True), 
                                               n_jobs=-1))
test_acc_svm = round(svmcla.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)
train_acc_svm = round(svmcla.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)

model2 = pd.DataFrame({
    'Model': ['SVM'],
    'Train Score': [train_acc_svm],
    'Test Score': [test_acc_svm*100]
})
model2.sort_values(by='Test Score', ascending=False)

#3. Random forest classification
from sklearn.ensemble import RandomForestClassifier

# We define the model
rfcla = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)
test_acc_rfcla = round(rfcla.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)
train_acc_rfcla = round(rfcla.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)

model3 = pd.DataFrame({
    'Model': ['Random Forest'],
    'Train Score': [train_acc_rfcla],
    'Test Score': [test_acc_rfcla*100]
})
model3.sort_values(by='Test Score', ascending=False)

#4. Decision tree classification

from sklearn.tree import DecisionTreeClassifier

# We define the model
dtcla =  DecisionTreeClassifier(random_state=9)
test_acc_dtcla = round(dtcla.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)*100, 2)
train_acc_dtcla = round(dtcla.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)

model4 = pd.DataFrame({
    'Model': ['Decision Tree'],
    'Train Score': [train_acc_dtcla],
    'Test Score': [test_acc_dtcla*100]
})
model4.sort_values(by='Test Score', ascending=False)

#5. K-Nearest Neighbor classification

from sklearn.neighbors import KNeighborsClassifier

# We define the model
knncla = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)

test_acc_knncla = round(knncla.fit((x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)
train_acc_knncla = round(knncla.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)

model5 = pd.DataFrame({
    'Model': ['KNN'],
    'Train Score': [train_acc_knncla],
    'Test Score': [test_acc_knncla*100]
})
model5.sort_values(by='Test Score', ascending=False)

#Comparison of classification techniques

model5 = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM','Random Forest','Decision Tree','KNN'],
    'Train Score': [train_acc_logregsk, train_acc_svm , train_acc_rfcla ,train_acc_dtcla, train_acc_knncla],
    'Test Score': [test_acc_logregsk, test_acc_svm , test_acc_rfcla ,test_acc_dtcla,test_acc_knncla ]
})
model5.sort_values(by='Test Score', ascending=False)
