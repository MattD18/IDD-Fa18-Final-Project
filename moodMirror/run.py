# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import cv2


from picamera import PiCamera
import requests
import time

import pygame

#initialize music
pygame.mixer.init()
happy = pygame.mixer.Sound('happy.wav')
sad = pygame.mixer.Sound('sadder.wav')
surprise = pygame.mixer.Sound('surprise.wav')
angry = pygame.mixer.Sound('angry.wav')
happy_state = 1
sad_state = 1
angry_state = 1
surprise_state = 1

#initialize facial recognition parameters
subscription_key = '4c33d2e569b7474aa1478ef390c210a4'
assert subscription_key
face_api_url = 'https://eastus.api.cognitive.microsoft.com/face/v1.0/detect'
headers = { 'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type':'application/octet-stream' }
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
}

#initialize person detector
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
person_counter = 0

#initialize net
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > .2:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            if CLASSES[idx] == 'person':
                person_counter += 1
    if person_counter >= 3:
        print('Snap!')
        cv2.imwrite('person.jpg', frame)
        cv2.imwrite('face.jpg', vs.read())
        with open('face.jpg', 'rb') as f:
            data = f.read()
            response = requests.post(face_api_url, params=params, headers=headers, data=data)
            faces = response.json()
            print(len(faces))
            if len(faces) > 0:
                print(faces[0]['faceAttributes']['emotion'])
                sad_level = faces[0]['faceAttributes']['emotion']['sadness']
                happy_level = faces[0]['faceAttributes']['emotion']['happiness']
                angry_level = faces[0]['faceAttributes']['emotion']['anger']
                surprise_level =faces[0]['faceAttributes']['emotion']['surprise']
                if sad_level > max(happy_level,angry_level,surprise_level,.15):
                    print('sad')
                    sad_state = 1
                    if angry_state == 1 or surprise_state == 1 or happy_state == 1:
                        happy_state = 0
                        angry_state = 0
                        surprise_state = 0
                        pygame.mixer.stop()
                        sad.play()
                elif happy_level > max(sad_level,angry_level,surprise_level,.25):
                    print('happy')
                    happy_state = 1
                    if angry_state == 1 or surprise_state == 1 or sad_state == 1:
                        sad_state = 0
                        angry_state = 0
                        surprise_state = 0
                        pygame.mixer.stop()
                        happy.play()
                elif angry_level > max(happy_level,sad_level,surprise_level,.15):
                    print('angry')
                    angry_state = 1
                    if happy_state == 1 or surprise_state == 1 or sad_state == 1:
                        happy_state = 0
                        surprise_state = 0
                        sad_state = 0
                        pygame.mixer.stop()
                        angry.play()
                elif surprise_level > max(happy_level,angry_level,sad_level,.15):
                    print('surprise')
                    surprise_state = 1
                    if sad_state == 1 or angry_state == 1 or happy_state == 1:
                        sad_state = 0
                        angry_state = 0
                        happy_state = 0
                        pygame.mixer.stop()
                        time.sleep(.4)
                        surprise.play()
                else:
                    print('neutral')

        person_counter = 0
    fps.update()
    time.sleep(.1)

# do a bit of cleanup
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
pygame.mixer.stop()
