# -*- coding: utf-8 -*-


from tensorflow.keras.layers import Input,Lambda,Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

train_data="/content/drive/MyDrive/Curl_Dataset/Dataset.zip (Unzipped Files)/Training"
validation_data="/content/drive/MyDrive/Curl_Dataset/Dataset.zip (Unzipped Files)/Validation"
img_size=[224,224]

import tensorflow
resnet_152v2=tensorflow.keras.applications.ResNet152V2(input_shape=img_size+[3],weights='imagenet',include_top=False)

for layer in resnet_152v2.layers:
  layer.trainable=False

folder="/content/drive/MyDrive/Curl_Dataset/Dataset.zip (Unzipped Files)/Training/*"
list(folder[:5])

x=Flatten()(resnet_152v2.output)

predictions=Dense(len(folder),activation='softmax')(x)
model=Model(inputs=resnet_152v2.input,outputs=predictions)

Image_gen=ImageDataGenerator(rescale=1./255,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

train_image_gen=Image_gen.flow_from_directory(train_data,target_size=(224,224),batch_size=40,class_mode='binary')

validation_data_scale=ImageDataGenerator(rescale=1./255)
validation_image_gen=validation_data_scale.flow_from_directory(validation_data,target_size=(224,224),batch_size=40,class_mode='binary')

model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_image_gen,epochs=10,verbose=1,validation_data=validation_image_gen)

model.save("assignment.h5")

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

modd=load_model("assignment.h5")

pred=modd.predict_generator(train_image_gen, steps=len(train_image_gen), verbose=1)
filenames=train_image_gen.filenames
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_image_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
results=pd.DataFrame({"file":filenames,"class_label":predictions})
results.to_csv("train_data.csv")

pred=modd.predict_generator(validation_image_gen, steps=len(validation_image_gen), verbose=1)
filenames=validation_image_gen.filenames
predicted_class_indices=np.argmax(pred,axis=1)
labels = (validation_image_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
results1=pd.DataFrame({"file":filenames,"class_label":predictions})
results1.to_csv("validation.csv")

test_data="/content/drive/MyDrive/Curl_Dataset/Dataset.zip (Unzipped Files)/Testing"

test_datagen=ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory(test_data,
                                                target_size=(224,224),
                                                batch_size=20,
                                                class_mode='binary',
                                                shuffle=False,
                                                )

pred=modd.predict_generator(test_generator, steps=len(test_generator), verbose=1)

filenames=test_generator.filenames
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_image_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
results1=pd.DataFrame({"file_path":filenames,"class_label":predictions})
results1.to_csv("test_set.csv")

valid_test_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.3)
valid_test_generator=valid_test_datagen.flow_from_directory(test_data,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                batch_size=20,
                                                class_mode='binary',
                                                subset='validation',
                                                shuffle=False,
                                                )

train_test_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.3)
train_test_generator=train_test_datagen.flow_from_directory(test_data,
                                                target_size=(224,224),
                                                color_mode='rgb',
                                                batch_size=20,
                                                class_mode='binary',
                                                subset='validation',
                                                shuffle=False,
                                                )

pred=modd.predict_generator(train_test_generator, steps=len(train_test_generator), verbose=1)

filenames=train_test_generator.filenames
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_image_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
results1=pd.DataFrame({"file_path":filenames,"class_label":predictions})
results1.to_csv("train_test_set.csv")

pred=modd.predict_generator(valid_test_generator, steps=len(valid_test_generator), verbose=1)

filenames=valid_test_generator.filenames
predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_image_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
results1=pd.DataFrame({"file_path":filenames,"class_label":predictions})
results1.to_csv("valid_test_set.csv")

