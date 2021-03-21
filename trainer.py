import os
import cv2
import glob
import numpy as np

imagePaths = []
def getImagesWithID(path):
        global imagePaths
        imagePaths = [os.path.join(p,f) for p in path for f in os.listdir(p)]
        images = []
        labels = []
        for imagePath in imagePaths:
                faceImg = cv2.imread(imagePath)
                label = int(os.path.split(imagePath)[-1].split('.')[1])
                images.append(prepare_image(faceImg))
                labels.append(label)
                cv2.imshow("training",faceImg)
                cv2.waitKey(10)
        print labels
        return labels, images

def prepare_image(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	return img

recognizer = cv2.createFisherFaceRecognizer()
path = glob.glob("dataset/*")
labels, images  = getImagesWithID(path)
recognizer.train(images, np.array(labels))
recognizer.save('recognition//trainingData.yml')
cv2.destroyAllWindows()
