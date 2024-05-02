#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

# Init camera
cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path='/data3/'
file_name = input("Enter the name of the person: ")

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    # Pick the last face (because it is the largest face according to area=w*h)
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Extract (crop out the required face)
        offset = 10
        face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        face_section = cv2.resize(face_section, (100, 100))
        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))
            
        # Display face section
        cv2.imshow("face section", face_section)

    cv2.imshow("frame", frame)

    # Wait for user input - 'q' then you will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF  # program wait for 1 ms
    if key_pressed == ord('q'):
        break

# Convert our face list into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save data into file
dataset_path = './data3/'
np.save(dataset_path + file_name + '.npy', face_data)
print("Data successfully saved at " + dataset_path + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




