# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:52:43 2020

@author: ningesh
"""



from flask import Flask, render_template, request, session, url_for, redirect, jsonify
import pymysql
import pandas as pd
import os
from imutils.video import VideoStream
app = Flask(__name__)
app.secret_key = 'random string'

#Database Connection
def dbConnection():
    connection = pymysql.connect(host="35.208.147.193", user="inbotics_student", password="inbotics_student", database="inbotics_studentdata")
    return connection

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:58:28 2019

@author: Sumit
"""
import os
import numpy as np
import cv2
import imutils
from collections import deque
import pickle

import urllib
import time
import datetime

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
#recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("./recognizers/face-trainner1.yml")





import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from time import time

string_pred_age = ['04-06', '07-08','09-11','12-19','20-27','28-35','36-45','46-60','61-75']
string_pred_gen = ['Female', 'Male']

# Load TFLite model and allocate tensors. Load Face Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

interpreter_age = tf.lite.Interpreter(model_path="AgeClass_best_06_02-16-02.tflite")
interpreter_age.allocate_tensors()

interpreter_gender = tf.lite.Interpreter(model_path="GenderClass_06_03-20-08.tflite")
interpreter_gender.allocate_tensors()

# # Get input and output tensors
input_details_age = interpreter_age.get_input_details()
output_details_age = interpreter_age.get_output_details()
input_shape_age = input_details_age[0]['shape']

input_details_gender = interpreter_gender.get_input_details()
output_details_gender = interpreter_gender.get_output_details()
input_shape_gender = input_details_gender[0]['shape']

input_im = None


import RPi.GPIO as GPIO
import time
relay = 8;

#Database Connection
vs = VideoStream(usePiCamera=False).start()
time.sleep(1.0)
import time,board,busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
import cv2
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000) # setup I2C
mlx = adafruit_mlx90640.MLX90640(i2c) # begin MLX90640 with I2C comm
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ # set refresh rate
# setup the figure for plotting
plt.ion() # enables interactive plotting
mlx_shape = (24,32)
frame = np.zeros((24*32,)) # setup array for storing all 768 temperatures
fig,ax = plt.subplots(figsize=(12,7))
therm1 = ax.imshow(np.zeros(mlx_shape),vmin=0,vmax=60) #start plot with zeros
cbar = fig.colorbar(therm1) # setup colorbar for temps
cbar.set_label('Temperature [$^{\circ}$C]',fontsize=14) # colorbar label
t_array = []


def dbClose():
    dbConnection().close()
    return
def sendemailtouser(usertoaddress,filetosend):   
    fromaddr = "beprojectcomputer@gmail.com"
    toaddr = usertoaddress#"ningesh1406@gmail.com"
   
    #instance of MIMEMultipart 
    msg = MIMEMultipart() 
  
    # storing the senders email address   
    msg['From'] = fromaddr 
  
    # storing the receivers email address  
    msg['To'] = toaddr 
  
    # storing the subject  
    msg['Subject'] = "alarm for unknown person"
  
    # string to store the body of the mail 
    body = "Unknown person"
  
    # attach the body with the msg instance 
    msg.attach(MIMEText(body, 'plain')) 
  
    # open the file to be sent  
    filename = filetosend
    attachment = open(filetosend, "rb") 
  
    # instance of MIMEBase and named as p 
    p = MIMEBase('application', 'octet-stream') 
  
    # To change the payload into encoded form 
    p.set_payload((attachment).read()) 
  
    # encode into base64 
    encoders.encode_base64(p) 
   
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename) 
  
    # attach the instance 'p' to instance 'msg' 
    msg.attach(p) 
  
    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587) 
  
    # start TLS for security 
    s.starttls() 
  
    # Authentication 
    s.login(fromaddr, "beproject1") 
  
    # Converts the Multipart msg into a string 
    text = msg.as_string() 
  
    # sending the mail 
    s.sendmail(fromaddr, toaddr, text) 
  
    # terminating the session 
    s.quit() 
def otpsendingfunction(mobile,msgtosend):
    authkey = "175606AVhvZO37X59c2613b"  # Your authentication key.
    mobiles = mobile  # Multiple mobiles numbers separated by comma.
    message = msgtosend#"unknown face detected in you camera"  # Your message to send.
    sender = "ALARMF"  # Sender ID,While using route4 sender id should be 6 characters long.
    route = "route4"  # Define route
    # Prepare you post parameters
    values = {
        'authkey': authkey,
        'mobiles': mobiles,
        'message': message,
        'sender': sender,
        'route': route
    }
    url = "http://api.msg91.com/api/sendhttp.php"  # API URL
    postdata = urllib.parse.urlencode(values).encode("utf-8")  # URL encoding the data here.
    req = urllib.request.Request(url, postdata)
    response = urllib.request.urlopen(req)
    output = response.read()  # Get Response
    print(output)

    

#close DB connection



@app.route('/index')
#@app.route('/')
def index():
    if 'user' in session:
        return render_template('home.html', user=session['user'], s=list)
    else:
        session['userloc']= request.args.get("location")
        locationis=session['userloc']
        print(locationis)
        return render_template('index.html')

@app.route('/startsurvilliencecam')
def startsurvilliencecam():
    if 'user' in session:
        val= request.args.get("thresh")
        
    return redirect(url_for('index'))



@app.route('/captureuserfaceandsavebyname',methods=["GET","POST"])
def captureuserfaceandsavebyname():
    firstFrame = None
    datasetpath='facedata//'
    if request.method == "POST":
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
        cnt=0
        usernamelist='ningesh123'
        usernamelist =request.form["username"]
        company_name = request.form["company_name"]
        start_date = request.form["start_date"]
        End_date = request.form["End_date"]
        technlogy_worked = request.form["technlogy_worked"]
    
        con = dbConnection()
        cursor = con.cursor()
        sql = "INSERT INTO user_company_information (username, company_name, start_date, End_date, technology_worked) VALUES (%s, %s, %s, %s, %s)"
        val = (usernamelist, company_name, start_date, End_date, technlogy_worked)
        cursor.execute(sql, val)
        con.commit()
        dbClose()
        usernamelist =datasetpath+ request.form["username"]
        if not os.path.exists(usernamelist):
            os.makedirs(usernamelist)
    
        while(cap.isOpened()):
            ret, frame1 = cap.read()
            
            if ret == True:
                width = cap.get(3)   # float
                height = cap.get(4)
                surface = width * height
                
            #cv2.imshow('frame with line drawn',gray)
                print(width)
                text = "Unoccupied"
                frame1 = imutils.resize(frame1, width=500)
                faces = face_cascade.detectMultiScale(frame1,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
            
    # show the frame and record if the user presses a key
            
         
         
                key = cv2.waitKey(1) & 0xFF
                for (x, y, w, h) in faces:
                    cv2.putText(frame1, str(cnt),  (x, y) , cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                    cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    rectangleframe=frame1[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(usernamelist , str(cnt)+'.jpg'), rectangleframe)
                    cnt=cnt+1
                    print(cnt)
                
               
            #cv2.imshow('frame with line drawn',frame)
                
            #cv2.imwrite('image'+str(cnt)+'.png',roi_color)
                cv2.imshow("Security Feed", frame1) 
            if cnt>25:
                cnt=0
                break
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break 
        cap.release()
        cv2.destroyAllWindows()
        return 'Success'
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes
@app.route('/recognitionofperson')
def recognitionofperson():
    train_dir = 'facedata/'
    val_dir = 'facedata/'
    con = dbConnection()
    cursor = con.cursor()
    namedir=os.listdir(train_dir)
    print(namedir)
    lengthofclasses=len(namedir)
    #vs = VideoStream(usePiCamera=False).start()
    #time.sleep(1.0)
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    emotion_model = Sequential()

    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(lengthofclasses, activation='softmax'))
# emotion_model.load_weights('emotion_model.h5')

    cv2.ocl.setUseOpenCL(False)
    #cap = cv2.VideoCapture(0)
    nameofuser=''
    a_dict = {}
    genders=[]
    ages=[]
    font = cv2.FONT_HERSHEY_PLAIN
    
    while True:
    # Find haar cascade to draw bounding box around face
        frame1 = vs.read()
        padding=20

        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

            
        for (x, y, w, h) in num_faces:
            roi_gray_frame = frame1[y:y + h, x:x + w]
            roi_gray_frame2 = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame2, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            namefound=namedir[maxindex]
            nameofuser=namefound
            print(namefound)
            input_im =roi_gray_frame# saved_image[y:y+h, x:x+w]
        
            if input_im is None:
                print("Nu a fost detectata nicio fata")
            else:
                input_im = cv2.resize(input_im, (224,224))
                input_im = input_im.astype('float')
                input_im = input_im / 255
                input_im = img_to_array(input_im)
                input_im = np.expand_dims(input_im, axis = 0)

            # Predict
                input_data = np.array(input_im, dtype=np.float32)
                interpreter_age.set_tensor(input_details_age[0]['index'], input_data)
                interpreter_age.invoke()
                interpreter_gender.set_tensor(input_details_gender[0]['index'], input_data)
                interpreter_gender.invoke()

                output_data_age = interpreter_age.get_tensor(output_details_age[0]['index'])
                output_data_gender = interpreter_gender.get_tensor(output_details_gender[0]['index'])
                index_pred_age = int(np.argmax(output_data_age))
                index_pred_gender = int(np.argmax(output_data_gender))
                prezic_age = string_pred_age[index_pred_age]
                prezic_gender = string_pred_gen[index_pred_gender]
                genders.append(prezic_gender)
                ages.append(prezic_age)
            cv2.putText(frame1, prezic_age + ', ' + prezic_gender, (x,y), font, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.rectangle(frame1, (x,y), (x+w,y+h), (255,255,255), 1)   
            if namefound in a_dict:
                a_dict[namefound] += 1
            else:
                a_dict[namefound] = 1
            result_count = cursor.execute('SELECT * FROM user_company_information WHERE username = %s',(namefound))
            print('SELECT * FROM user_company_information WHERE username = %s',(namefound))
            res = cursor.fetchone()
            #print(res)
            userinfo=''
            nameofuser=''
            if result_count > 0:
                print('inside')
                print(result_count)
                nameofuser=namefound
                userinfo =namefound+'\n'+ str(res[0])+'\n'+ res[1]+'\n'+ res[2]+'\n'+ res[3]+'\n'+ res[4]+'\n'+ res[5]
            #cv2.putText(frame,namedir[maxindex]+str(maxindex), (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            y0, dy = 50, 50
            
            
            
           
            
        #time.sleep(2)
        
        
        
            t1 = time.monotonic()
            mlx.getFrame(frame) # read MLX temperatures into frame var
            print('Average MLX90640 Temperature: {0:2.1f}C ({1:2.1f}F)'.\
                  format(np.mean(frame),(((9.0/5.0)*np.mean(frame))+32.0)))
            tempobt=np.mean((((9.0/5.0)*np.mean(frame))+32.0))
            data_array = (np.reshape(frame,mlx_shape)) # reshape to 24x32
            #print(data_array)
            therm1.set_data(np.fliplr(data_array)) # flip left to right
            therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array)) # set bounds
            cbar.update_normal(therm1) # update colorbar range
            plt.pause(0.001) # required
            fig.savefig('mlx90640_test_fliplr.png',dpi=300,facecolor='#FCFCFC',
                        bbox_inches='tight') # comment out to speed up
            t_array.append(time.monotonic()-t1)
            #print('Sample Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))
            img=cv2.imread('mlx90640_test_fliplr.png')
            dim=(400,400)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            #tempobt=35
            cv2.putText(frame1,str(tempobt), (150,150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,0))
            cv2.imshow('image',resized)
            
            
            
            cursor = con.cursor()
            result_count2=cursor.execute('SELECT * FROM temperatureinfo WHERE usernamefound = %s', (nameofuser))
            res = cursor.fetchone()
            agemax1=max(ages)
            gendermax1=max(genders)
            ts = time.time()
            if result_count2>0:
                cursor.execute('delete FROM temperatureinfo WHERE usernamefound = %s', (nameofuser))
            
                temperature=str(tempobt)
                filename='static/headimages/'+str(ts)+'.jpg'
                cv2.imwrite(filename,roi_gray_frame)
                sql = "INSERT INTO temperatureinfo (filename, usernamefound, temperature,age,gender) VALUES (%s, %s, %s, %s, %s)"
                val = (filename, nameofuser, temperature,agemax1,gendermax1)
                cursor.execute(sql, val)
                con.commit()
            else:
                temperature=str(tempobt)
                filename='static/headimages/'+str(ts)+'.jpg'
                cv2.imwrite(filename,roi_gray_frame)
                sql = "INSERT INTO temperatureinfo (filename, usernamefound, temperature,age,gender) VALUES (%s, %s, %s, %s, %s)"
                val = (filename, nameofuser, temperature,agemax1,gendermax1)
                cursor.execute(sql, val)
                con.commit()
            #dbClose()
            
            for i, line in enumerate(userinfo.split('\n')):
                y = y0 + i*dy
                cv2.putText(frame1, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2, cv2.LINE_AA)
            #cv2.putText(frame,userinfo, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print('in video')          
        cv2.imshow('Video', cv2.resize(frame1,(1200,860),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    dbClose()
    #cap.release()
    cv2.destroyAllWindows()
    vs.stop()
    Keymax = max(a_dict, key=a_dict.get) 
    print(Keymax) 
    print('dict is ',a_dict)
    nameofuser=Keymax
    cursor2 = con.cursor()
        #cursor2.execute('SELECT item, quantity, metrics, calorie, carbohydrates, protein,fat  FROM calorie1 ORDER BY RAND() limit 5')
        #cursor2.execute("SELECT item, quantity, metrics, calorie, carbohydrates, protein,fat  FROM calorie1new where vegornonveg = %s and at = 'd' and type='"+str(opfor)+"' and disease not like '"+diseaseobt+"' ORDER BY RAND()", (vegornonveg,))
    cursor2.execute("select * from user_company_information where username='"+nameofuser+"'")
    print("select * from user_company_information where username='"+nameofuser+"'")
    res11 = cursor2.fetchall()
    li2=[]
    agemax=max(ages)
    gendermax=max(genders)
    for a in res11:
        li2.append(a)
        
    #li2.append(agemax)
    #li2.append(gendermax)
    #print('list is',li2)
    return render_template('output.html',data=li2,gender=gendermax,age=agemax)
    
    
@app.route('/traningdataset')
def traningdataset():
    train_dir = 'facedata/'
    val_dir = 'facedata/'

    namedir=os.listdir(train_dir)
    print(namedir)
    lengthofclasses=len(namedir)

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=64,
            color_mode="grayscale",
            class_mode='categorical')

    emotion_model = Sequential()

    emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
    emotion_model.add(Dropout(0.25))

    emotion_model.add(Flatten())
    emotion_model.add(Dense(1024, activation='relu'))
    emotion_model.add(Dropout(0.5))
    emotion_model.add(Dense(lengthofclasses, activation='softmax'))
# emotion_model.load_weights('emotion_model.h5')

    cv2.ocl.setUseOpenCL(False)

#emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


    emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    emotion_model_info = emotion_model.fit_generator(
            train_generator,
            steps_per_epoch=28709 // 64,
            epochs=1,
            validation_data=validation_generator,
            validation_steps=7178 // 64)
    emotion_model.save_weights('face_model.h5')
    
    return 'success'



@app.route('/home')
def home():
    if 'user' in session:
        return render_template('home.html', user=session['user'], s=list)
    return redirect(url_for('index'))


@app.route('/login', methods=["GET","POST"])
def login():
    msg = ''
    
    if request.method == "POST":
        # session.pop('user',None)
        mobno = request.form.get("mobile")
        password = request.form.get("pas")
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM userdetails WHERE mobileno = %s AND password = %s',(mobno, password))
        res = cursor.fetchone()
        print(res)
        if result_count > 0:
            print(result_count)
            session['user'] = mobno
            session['uid'] = res[0]
            
            return redirect(url_for('home'))
        else:
            print(result_count)
            msg = 'Incorrect username/password!'
            return render_template('login.html')
    return render_template('login.html')

@app.route('/register', methods=["GET","POST"])
def register():
    print("register")
    if request.method == "POST":
        try:
            name = request.form.get("name")
            address = request.form.get("address")
            mailid = request.form.get("mailid")
            mobile = request.form.get("mobile")
            pass1 = request.form.get("pass1")
            con = dbConnection()
            cursor = con.cursor()
            cursor.execute('SELECT * FROM userdetails WHERE mobile = %s', (mobile))
            res = cursor.fetchone()
            if not res:
                sql = "INSERT INTO userdetails (name, address, email, mobile, password) VALUES (%s, %s, %s, %s, %s)"
                val = (name, address, mailid, mobile, pass1)
                cursor.execute(sql, val)
                con.commit()
                
                sql1 = "INSERT INTO readingcount (uid, ht_count, toi_count, ie_count) VALUES (%s, %s, %s, %s)"
                val1 = (mobile,int(0),int(0),int(0))
                cursor.execute(sql1, val1)
                con.commit()
                status= "success"
                return redirect(url_for('index'))
            else:
                status = "Already available"
            return status
        except Exception as inst:
            print(inst)
            print("Exception occured at user registration")
            return redirect(url_for('index'))
        finally:
            dbClose()
    return render_template('register.html')



#logout code
@app.route('/logout')
def logout():
    session.pop('user')
    return redirect(url_for('index'))



#---------------------------------

@app.route('/index1')
@app.route('/')
def hello():
    return render_template('indexx.html')


@app.route('/state.html')
def state():
    return render_template('state.html')




@app.route('/viewhightemepraturedata', methods=["GET","POST"])
def viewhightemepraturedata():
    msg = ''
    
    if request.method == "POST":
        # session.pop('user',None)
        #mobno = request.form.get("mobile")
        
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM temperatureinfo where temperature>100')
        res = cursor.fetchall()
        print(res)
    if request.method == "GET":
        # session.pop('user',None)
        #mobno = request.form.get("mobile")
        
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM temperatureinfo where temperature>100')
        res = cursor.fetchall()
        print(res)
        
        
    return render_template('tempoutput.html',data=res)

@app.route('/viewalltemepraturedata', methods=["GET","POST"])
def viewalltemepraturedata():
    msg = ''
    
    if request.method == "POST":
        # session.pop('user',None)
        #mobno = request.form.get("mobile")
        
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM temperatureinfo')
        res = cursor.fetchall()
        print(res)
    if request.method == "GET":
        # session.pop('user',None)
        #mobno = request.form.get("mobile")
        
        con = dbConnection()
        cursor = con.cursor()
        result_count = cursor.execute('SELECT * FROM temperatureinfo')
        res = cursor.fetchall()
        print(res)    
        
    return render_template('tempoutput.html',data=res)




if __name__ == '__main__':
    app.run('0.0.0.0')
    #app.run()
