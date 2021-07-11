# Facial emotion recognition

## INSTALL requirement
 - numpy
 - pandas
 - tensorflow
 - keras
 - opencv2
 - mtcnn
 - keras_vggface
 - keras_applications
&nbsp;
&nbsp;
## Dataset
 - #### Fer2013
 ````
 https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
 ````
&nbsp;
## RUN
 ### For training custom model
 - #### Custom model
 ````
 python training_custom_model.py
 ````
 
 - #### VGGFace model
 Read more in Emotion_tranfer_learning_VGG.ipynb
 &nbsp;
 &nbsp;
 ### For practical application
 - #### Video with custom model
 ````
 python video_detect_haarcascade_custom_model.py --video <path input video> --save <path output detected video> --fps <fps to detect> -- model <path model> -- weight <path weight>
 ````
 
 - #### Camera with custom model
 ````
 python camera_testing_haarcascade_custom_model.py
 ````
 
 - #### Video with VGG model
 ````
 python video_detect_mtcnn_vgg_model.py --video <path input video> --save <path output detected video> --fps <fps to detect> -- weight <path weight>
 ````
&nbsp;
## Performance evaluation of the model
 ### Custom model - 63% accuracy
 ![alt text](training_custom_model.png?raw=true)
 ### VGG model - 85% accuracy (best weight in 32nd epoch)
 ![alt text](training_vgg_model.png?raw=true)

 
