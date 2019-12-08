# Bird-monitoring
A pre-processing technique to automatically locate bird image frames from a long video data

# Task

Given an input wild life video recording, the task is to output a processed video that only have bird image frames

# Method

1. Collected wild life recordings from bird activity zones using Wingscapes Birdcam Pro camera.
2. 60% of the video data is converted into image frames. Manually annotate bird image frames and background frames.
3. In our dataset we have 4 different bird species and 6 different backgrounds.
4. Train a simple CNN binary classifier to classifiy a given image into bird or non-bird (background).
5. This trained model is used to predict bird activity frames in the input video recording during testing.
6. The bird activity frames detected by the model is converted back to video format which is the processed output video.

# Description

    classifier.py - CNN classifier to predict bird frames
    video_frame.py - main code for video preprocessing
    model.h5 - trained model
    
 [One example of original video and processed video](https://drive.google.com/open?id=1B2y5jic5VSlAkqxGmo_nTrz5dMsr5OUz) 

# Inference and Future work

The CNN binary classifier can learn the properties of bird parts and is good to discriminate between bird image frames and backgrounds. In case of extended backgrounds and bird species from other recordings (eg. far field recordings) it is easier to finetune this model on new set of birds and backgrounds.
