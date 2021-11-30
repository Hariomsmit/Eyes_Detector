import cv2
cascade_classifier = cv2.CascadeClassifier('haarcascades/eye.xml')
OM = cv2.VideoCapture(0)
while True:
    ret, frame = OM.read()
    lens = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(lens,scaleFactor=1.3,minNeighbors=5)
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('Eyes Detection Author:Hari Om',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
OM.release()
cv2.destroyAllWindows()