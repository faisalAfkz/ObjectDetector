import cv2

#img = cv2.imread('rand.JPG')
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)


classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean ((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0))
            cv2.putText(img,classNames[classId-1] +" "+ str(int(confidence*100))+"%",(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))



    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
