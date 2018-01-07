import os
import numpy as np

from skimage import io, transform
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

EPOCHS=20
IMG_SIZE=256
DATA_DIR='dataset'
TEST_DATA_FRACTION=0.2

def transform_img(img):
    return transform.resize(img,(IMG_SIZE,IMG_SIZE,img.shape[2]))

    

def load_data():
    files=os.listdir(DATA_DIR)
    train_data=[]
    train_labels=[]
    
    for file in files:
        if file[-4:]=='jpeg':
            transformed_img=transform_img(io.imread(DATA_DIR+'/'+file))
            train_data.append(transformed_img)
            label_file=file[:-5]+'.txt'
            
            with open(DATA_DIR+'/'+label_file) as f:
                content=f.readlines()
                label=int(float(content[0]))
                l=[0,0]
                l[label]=1
                train_labels.append(l)
                
    return np.array(train_data),np.array(train_labels)

def cnn():
    classifier=Sequential()

    classifier.add(Convolution2D(filters=16,kernel_size=(3,3),input_shape=(IMG_SIZE,IMG_SIZE,3),activation='relu'))    
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(IMG_SIZE,IMG_SIZE,3),activation='relu'))    
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Flatten())
    classifier.add(Dense(units=2,activation='softmax'))
    
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    return classifier


data,labels=load_data()

from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = TEST_DATA_FRACTION)

idx = np.random.permutation(train_data.shape[0])
classifier = cnn()
classifier.fit(train_data[idx], train_labels[idx],verbose=2, epochs=EPOCHS)

preds = np.argmax(classifier.predict(test_data), axis=1)
test_labels = np.argmax(test_labels, axis=1)

cm = confusion_matrix(test_labels, preds)
accuracy=accuracy_score(test_labels,preds)
precision=precision_score(test_labels,preds)
recall=recall_score(test_labels,preds)
f1=f1_score(test_labels,preds)


