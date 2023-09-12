import os
import cv2 as cv
import numpy as np
from  tensorflow import keras


classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
yolo_net = cv.dnn.readNet('yolov3.weights' , 'yolov3.cfg')


layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

cap = cv.VideoCapture('Test Videos/1.mp4')
# if you want it for live video(on youre webcam):
# cap = cv.VideoCapture(0)

new_width = 20000 
new_height = 20000
cap.set(cv.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, new_height)

frame_number = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        img = cv.resize(frame, None, fx=0.8, fy=0.8)
        height, width, channels = img.shape
        blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outs = yolo_net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                # if you want you can choose what exacly detect like person only
                #if classes[class_ids[i]] == "person":
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = (0,0,255)
                    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv.putText(img, label, (x, y + 30), font, 2, color, 2 )
                    
                    # saving photo's when detect
                    if label == "person":
                        # frame_file_name = 'Faces\frame_{frame_number}.jpg'
                        with open('Faces\person_{}.png'.format(frame_number), "wb") as f:
                            cv.imwrite('Faces\person_{}.png'.format(frame_number), img)
    
        cv.imshow('Frame', img)
        if cv.waitKey(12) & 0xFF == ord('q'):
            break
        
        frame_number += 1
        
cap.release()
cv.destroyAllWindows()
