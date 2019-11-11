import warnings
warnings.filterwarnings('ignore')

import os

# IF CPU DESIRED
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # USE CPU memory
os.environ["CUDA_VISIBLE_DEVICES"]="" # No GPUS

# IF GPU DESIRED 
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" #GPUS ID
import numpy as np
import glob
import cv2
import time 
import pandas as pd 
import matplotlib.pyplot as plt

from keras.models import load_model,Sequential
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
from keras.utils import multi_gpu_model, np_utils
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import  Adam,SGD
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint,TensorBoard,LearningRateScheduler

import scipy




def crop_to_square(image): 
    heigh, width, _ = image.shape
    if(width > heigh):
        difference = width-heigh
        padd = int(difference/2.0)
        return image[25:heigh-25,(padd+25):width-(padd+25),:]
    elif(heigh>width):
        difference = heigh-width
        padd = int(difference/2.0)
        return image[(padd+25):heigh-(padd+25),25:width-25,:]
    return image


def prepare_img_for_prediction(img_path):
    # load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = crop_to_square(img)
    
    img  = img /  255.0  # normalise
    img = cv2.resize(img,(224,224)) # resize
    img_tensor = np.expand_dims(img,axis=0) # attach the batch dim

    return img_tensor



def calculate_accuaracy_over_each_class(DataPath=None, model = None):
    max_time = 0.0
    classes = os.listdir(DataPath) # get all classes names from directories label
    accuracy_per_class = {}
    print("\n Found ",classes)
    os.system("mkdir missed")
    os.system("rm -rf missed/*")
    for xclass in classes: # loop over folders/classes

        os.system("mkdir missed/"+xclass+"_test_missed")
        
        true_prediction_counter = 0
        false_prediction_counter = 0

        imgs_file_list = glob.glob(os.path.join(os.path.join(DataPath, xclass),'*')) # load all images in a given directory and process them 
        
        for img_path in imgs_file_list:
            
            img_tensor = prepare_img_for_prediction(img_path)
            # predict
            start_time = time.time()
            prediction = model.predict(img_tensor)
            needed_time = time.time()-start_time

            if(needed_time> max_time):
                max_time = needed_time
            
            predicted_label = classes_names[np.argmax(prediction)]
            
            if(predicted_label == xclass):
                true_prediction_counter +=1
            else:
                os.system("cp "+img_path+" missed/"+xclass+"_test_missed/"+predicted_label+"_"+str(false_prediction_counter)+".jpg")
                false_prediction_counter +=1

        if(false_prediction_counter==0):
            os.system("rm -rf missed/"+xclass+"_test_missed")


        accuracy = round(float(true_prediction_counter)/float(true_prediction_counter+false_prediction_counter),4)

        accuracy_per_class[xclass]={'hit':true_prediction_counter, 'miss':false_prediction_counter, 'accuracy':accuracy}
        
        print("\n====================================================")
        print(accuracy_per_class)

    return max_time




def get_classes_and_data_count_per_class_from_generator(generator):

    classes = generator.classes # array[0,0,0,...,1,1,1,....,2,2,2,..]# get all data classes ids 
    classes, data_count = np.unique(classes, return_counts=True) # get the count of unique values in the data list 
    # get data size according to unique class  
    data_per_class = list(dict(zip(classes, data_count)).values())

    return classes, data_per_class




def create_top_layers(model,drop_rate=0.1,activation='relu',top_layers_dims=None):


    newModel = Sequential()
    newModel.add(model)
    newModel.add(Flatten(input_shape=model.layers[len(model.layers)-1].output_shape))
    
    #top_layers_dims=[528,132],

    for dim in top_layers_dims:
            newModel.add(Dense(dim))
            newModel.add(BatchNormalization())
            if(activation=='relu'):
                newModel.add(Activation('relu'))
            else:
                newModel.add(LeakyReLU(alpha=0.1))
            
            newModel.add(Dropout(rate=drop_rate))    
    
    newModel.add(Dense(nb_classes))
    newModel.add(Activation('softmax'))

    return newModel





def calc_classweight(train_data_per_class):
    weight_dict = {}
    
    MaxExamplesCount = np.max(list(train_data_per_class.values()))
    
    print("MaxExamplesCount = ", MaxExamplesCount)

    for key in train_data_per_class.keys():
        weight_dict[key]=float(train_data_per_class[key])/float(MaxExamplesCount)

    return weight_dict




def load_pretrained_model(model_path=None):
    start_loading = time.time()
    model = load_model(model_path)
    load_time = time.time()-start_loading
    print("==>> model loaded in "+str(load_time)+" sec")
    return model



def _evaluate_generator(test_generator=None):
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    predict = model.evaluate_generator(test_generator,steps = nb_samples,verbose=1)
    print("Test Loss:", predict[0]," *** " " Test Accuaracy:", predict[1])
    return predict



def create_input_tensor_shape(dataformat):
    input_tensor_shape = None
    if dataformat == 'channels_last': # creat input tensor with shape [h, w, c]
        input_tensor_shape = Input(shape=(img_height,img_width,img_channel)) 
    else: # create input tensor with shape = [c, h, w]
        input_tensor_shape = Input(shape=(img_channel,img_height,img_width))

    return input_tensor_shape



def extract_singleModel_from_multiGpusModel(multiGpuModel_path, verbose=False):

    parrallel_model = load_pretrained_model(model_path=multiGpuModel_path)

    for layer in parrallel_model.layers:
        if 'lamda' in layer.name:
            
            single_model = parrallel_model.layers[-2]
            if(verbose):
                print("model before extraction summary")
                parrallel_model.summary()

                print("model after extraction summary")
                single_model.summary()
            
            return single_model

    single_model = parrallel_model
    return single_model





def get_ClassActivationMapModel(model):

    # get softmax (final layer [predictions])  weights
    softmax_weights = model.layers[-1].get_weights()[0] # 0 index for weights , 1 index for baises
    # extract final conv output, and prediction output
    CAM_model = Model(inputs=model.input, outputs=(model.layers[-4].output, model.layers[-1].output)) 
    
    return CAM_model, softmax_weights




def predict_CAM(img_as_tensor, cam_model, weights):
    # get filtered images from convolutional output + model prediction vector
    last_conv_output, pred_vec = cam_model.predict(img)
    # change dimensions of last convolutional outpu to 7 x 7 x 512 from 1x7x7x512
    last_conv_output = np.squeeze(last_conv_output) 
    # get model's prediction (number between 0 and 999, inclusive)
    pred = np.argmax(pred_vec)
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) # dim: height x width x channels
    # get AMP layer weights
    weights = weights[:, pred] # dim: (512,) 
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224, 512)), weights).reshape(224,224) # dim: 224 x 224
    # return class activation map
    return final_output, pred