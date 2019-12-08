#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:33:53 2018

@author: arjun
"""
#### video to image frames ######################################################################################################

import cv2
import os
import numpy as np
from keras.models import load_model
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras.preprocessing import image

# original video to test
video_file = '../WSCT1220.MOV'
# path to save the image frames of the test video data
path_to_save = '.../frames'


# video to image frames
def video_to_frames(video_file, path_to_save):
   vidcap = cv2.VideoCapture(video_file) # read the video data
   success,image = vidcap.read()
   count = 0
   success = True
   while success:
     cv2.imwrite(os.path.join(path_to_save,"{:d}.jpg".format(count)), image)     # save frame as JPEG file
     success,image = vidcap.read()
     print 'Read a new frame: ', success
     count += 1
   print('{} images are extacted in {}.'.format(count,path_to_save))

############################################################################################################################

video_to_frames(video_file, path_to_save)

############################################################################################################################
# creating image filelist

image_file = []

for file in os.listdir(path_to_save):
   image_file.append(file)


image_file = np.array(image_file)

#############################################################################################################################

input_path = path_to_save
image_data = []

# image resize
def read_resize_images(input_path, image_file, image_data):
   for i in range(len(image_file)):
      img = image.load_img(os.path.join(input_path,image_file[i]),target_size=(224,224))
      img = image.img_to_array(img)
      image_data.append(img)
      

read_resize_images(input_path, image_file, image_data)      
image_data=np.array(image_data)
      

##################################################################################################################################
# load trained model

model = load_model("../images_to_video_model.h5")
#adada = Adam(lr=0.001, decay = 1e-6)
#model.compile(loss='binary_crossentropy', optimizer=adada, metrics=['accuracy'])
class_label = model.predict_classes(image_data, batch_size=1, verbose=0)
class_label = np.array(class_label,dtype=int)

new_image_file = []

for i in range(len(class_label)):
   if class_label[i] == 1:
     new_image_file.append(image_file[i].replace(".jpg", ""))
     

new_image_file = np.array(new_image_file,dtype=int)
##################################################################################################################################

input_path = path_to_save
# output video path
out_path =  '.../processed.avi'
# path to save bird image frames
out_path_image = '../vid_img'
# frame rate
fps = 30.0

def frames_to_video(input_path,out_path,out_path_image,fps):
   image_array = []
   sorted_image_list = sorted(new_image_file)
#   print(sorted_image_list)
   for i in range(len(sorted_image_list)):
       img = cv2.imread(os.path.join(input_path,str(sorted_image_list[i])+'.jpg'))
       cv2.imwrite(os.path.join(out_path_image, str(sorted_image_list[i])+'.jpg'), img)
       size =  (img.shape[1],img.shape[0])
       image_array.append(img)
#   fourcc = cv2.VideoWriter_fourcc(*'XVID') # for opencv version >=3
   fourcc = cv2.cv.CV_FOURCC(*'XVID') # for opencv version 2 (#change the codec format accordingly, depending on the version of cv2)
   out = cv2.VideoWriter(out_path, fourcc, fps, size) 
   for i in range(len(image_array)):
       out.write(image_array[i])
#       cv2.imshow('video',image_array[i])
#      if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
#         break
      
   out.release()
#   cv2.destroyAllWindows()

####################################################################################################################################

frames_to_video(input_path,out_path,out_path_image,fps)

######################################################################################################################################
