from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import scipy
import tensorflow as tf
import align.detect_face



# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
import cv2
from keras.preprocessing import image
import time
import json

# Define a flask app
app = Flask(__name__)

# opencv 
#face_cascade = cv2.CascadeClassifier('/home/topica/anaconda3/envs/workspace/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

# facenet 
image_size=160
margin= 44
gpu_memory_fraction=1.0

def load_and_align_data(image_path, image_size,margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    img = scipy.misc.imread(os.path.expanduser(image_path))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if (len(bounding_boxes)==0):
        bb=0
        have_face = 0
    else:
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2 - bb[0], img_size[1])
        bb[3] = np.minimum(det[3]+margin/2 - bb[1], img_size[0])
        have_face = 1
    return bb,have_face

#
from keras.models import model_from_json
# model = model_from_json(open("/home/thaovu/tensorflow-101/model/facial_expression_model_structure.json", "r").read())
# model.load_weights('/home/thaovu/tensorflow-101/model/facial_expression_model_weights.h5') #load weights
model = model_from_json(open("/home/thaovu/keras-flask-deploy-webapp/models/model_4layer_2_2_pool.json", "r").read())
model.load_weights('/home/thaovu/keras-flask-deploy-webapp/models/model_4layer_2_2_pool.h5') #load weights


# Model saved with Keras model.save()
#MODEL_PATH = 'models/model_4layer_2_2_pool.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:8000/')


def model_predict(img_path, model):


    
# opencv 
#    img = cv2.imread(img_path)
#    img = cv2.resize(img, (640, 360))
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#    preds = []
#    for (x,y,w,h) in faces:
#        #if w > 130: #trick: ignore small faces
#        #cv2.rectangle(img,(x,y),(x+w,y+h),(64,64,64),2) #highlight detected face

#        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
#        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
#        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

#        img_pixels = image.img_to_array(detected_face)
#        img_pixels = np.expand_dims(img_pixels, axis = 0)

#        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        #------------------------------
#        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
#        preds.append(predictions)  

#facenet 
    img = cv2.imread(img_path)
    detect_face, have_face= load_and_align_data(img_path,image_size,margin,gpu_memory_fraction)
    preds = []
    detect = []
    if (have_face==0):
        print ('Cannot find any faces!!!')
    else:
        detect_face = np.reshape(detect_face,(-1,4)) 
        
        for (x,y,w,h) in detect_face:
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            crop_face = detected_face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            #------------------------------

            predictions = model.predict(img_pixels) #store probabilities of 7 expressions
            preds.append(predictions)
            detect.append(detect_face)
    return preds,have_face , crop_face, detect


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    data = {"success": False}
    if request.method == 'POST':
        # # Get the file from post request
        # f = request.files['file']
        #
        # # Save the file to ./uploads
        # basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)
        #
        # # Make prediction
        # emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        # preds, have_face, crop_face = model_predict(file_path, model)
        # if (have_face == 0):
        #     result = " Can't not find any face :("
        # else:
        #     emotion_array = []
        #     # Process your result for human
        #     max_index = np.argmax(preds[0])
        #     file_path_crop = os.path.join(
        #     basepath, 'dataset', emotions[max_index], secure_filename(f.filename))
        #     cv2.imwrite(file_path_crop, crop_face)
        #     result = " "
        #     for j in range(len(preds)):
        #             emotion = ""
        #             a = ""
        #             for i in range(len(preds[j][0])):
        #                 emotion = "%s %s%s" % (emotions[i], round(preds[j][0][i]*100, 2), '%')
        #                 emotion_array.append(emotion)
        #             a = "face%s: %s %s %s %s %s %s %s" % (str(j+1), emotion_array[0], emotion_array[1],emotion_array[2],emotion_array[3],emotion_array[4],emotion_array[5],emotion_array[6])
        #             result = result + a + " "
        # return result
        f = request.files["file"]
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        data["predictions"] = []
        data["result"] = []
        data["predictions"] = []
        preds, have_face, crop_face,detect = model_predict(file_path, model)

        if (have_face != 0):

            for j in range(len(preds)):
                max_index = np.argmax(preds[j][0])
                file_path_crop = os.path.join(
                    basepath, 'dataset', emotions[max_index], secure_filename(f.filename))
                cv2.imwrite(file_path_crop, crop_face)
                data["position"] = [{"x": float(detect[j][0][0]), "y": float(detect[j][0][1]),
                                     "w": float(detect[j][0][2]), "h": float(detect[j][0][3])}]
                data["result"] = {"result": emotions[max_index]}
                for i in range(len(emotions)):
                    r = {"label": emotions[i], "probability": round(preds[j][0][i] * 100, 2)}
                    data["predictions"].append(r)
        data["success"] = True
        import json
        data_name = f.filename.split('.')[0] + ".json"
        # os.path.join(basepath,'result', data_name)
        with open(data_name, 'w') as outfile:
            json.dump(data, outfile)
        return jsonify(data)
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 8000), app)
    http_server.serve_forever()
