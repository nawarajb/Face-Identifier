from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import time
import sqlite3

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(320, 240))
time.sleep(0.1)

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
rec = cv2.createFisherFaceRecognizer()
rec.load("/home/pi/Desktop/minor2/recognition/trainingData.yml")
label = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)

def getProfile(id):
        conn = sqlite3.connect("face_base.db")
        cmd = "SELECT * FROM Faces WHERE ID=" + str(id)
        cursor = conn.execute(cmd)
        profile = None
        for row in cursor:
                profile = row
        conn.close()
        return profile

def prepare_image(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	return img

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    img = frame.array
    faces = detector.detectMultiScale(
		img,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(50, 50)
	)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        img1 = img[y:y+h, x:x+w]
        r = 200 / img1.shape[1]
        img2 = cv2.resize(img1, (200, 200), interpolation = cv2.INTER_AREA)
        final_img = prepare_image(img2)
        label,dist=rec.predict(final_img)
        profile = getProfile(label)
        if(profile!=None):
                cv2.cv.PutText(cv2.cv.fromarray(img),profile[1],(x,y+h+30),font,255)
                cv2.cv.PutText(cv2.cv.fromarray(img),profile[2],(x,y+h+60),font,255)
                cv2.cv.PutText(cv2.cv.fromarray(img),profile[3],(x,y+h+90),font,255)
    cv2.imshow("Face", img)
    rawCapture.truncate(0)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
