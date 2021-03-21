from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import time
import os
import sqlite3

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 27
rawCapture = PiRGBArray(camera, size=(320, 240))
time.sleep(0.1)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def insert_update(*args):
    conn = sqlite3.connect("face_base.db")
    cmd = "SELECT * FROM Faces WHERE ID=" + str(args[0])
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if(isRecordExist == 1):
        cmd = "UPDATE Faces(ID,Name,Age,Criminal Records) Values("+str(args[0])+','+str(args[1])+','+str(args[2])+','+str(args[3])+')'+"WHERE ID="+str(args[0])
    else:
        cmd="INSERT INTO Faces(ID,Name,Age,Criminal Records) Values("+str(args[0])+','+str(args[1])+','+str(args[2])+','+str(args[3])+')'
    conn.execute(cmd)
    conn.commit()
    conn.close()
    
Id = raw_input("enter id:")
Name = raw_input("enter name:")
Age = raw_input("enter age:")
Gender = raw_input("enter gender:")
Criminal_Records = raw_input("enter criminal records:")
insert_update(Id,Name,Age,Criminal_Records)
sampleNum = 0
path = "User" + Id
os.chdir("dataset")
os.mkdir(path)
os.chdir(path)

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    img = frame.array
    faces = detector.detectMultiScale(
		img,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(50, 50)
	)
    for (x,y,w,h) in faces:
        #incrementing sample number
        sampleNum=sampleNum+1

        #saving the captured face in the dataset folder
        img1 = img[y:y+h, x:x+w]
        r = 200 / img1.shape[1]
        img2 = cv2.resize(img1, (200, 200), interpolation = cv2.INTER_AREA)
        cv2.imwrite("User."+Id +'.'+ str(sampleNum) + ".png", img2)
    rawCapture.truncate(0)
    if sampleNum>10:
        break

