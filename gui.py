import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with open('model_a.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights('model_weights.weights.h5')

emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

cap = cv2.VideoCapture(0)   

smoothed_prediction = None
smoothing_factor = 0.5  
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        
        if smoothed_prediction is None:
            smoothed_prediction = maxindex
        else:
            smoothed_prediction = int(smoothing_factor * maxindex + (1 - smoothing_factor) * smoothed_prediction)
        
        cv2.putText(frame, emotion_dict[smoothed_prediction], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
 
    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()