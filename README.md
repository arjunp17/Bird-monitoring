# Bird-monitoring
Towards the development of a pre-processing technique to refine wildlife video recordings.

# Task
The cameras deployed for bird-monitoring has to function for prolonged hours. A huge manual effort is required to pre-process the recordings prior to use the data for bird classification and detection problems. To automate the pre-processing phase, I have implemented a simple convolutional neural network (CNN) based image classification model to differentiate bird-activity frames from background frames. Given an input wildlife video recording, the task is to output a processed video that contains only bird activity frames.

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
    
 [One example of original video and processed video](https://drive.google.com/open?id=1PY21B81pWQzM4hRm3X5DbfvgkWEe0KYC) 

# Inference and Future work

Inferences: The CNN based classifier can learn the properties of bird parts and is good enough to discriminate between bird image frames and backgrounds. In case of extended backgrounds and bird species from different recordingconditions (eg. far-field recordings), it is easier to finetune this model on a new set of birds and backgrounds.
