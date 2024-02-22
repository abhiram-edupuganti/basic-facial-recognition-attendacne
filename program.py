
from datetime import datetime
from aiohttp import WSCloseCode, WSMessage
import cv2
import os
import face_recognition
import numpy as np
import csv


rdj=face_recognition.load_image_file("photos/rdj.jpeg")
rdj=face_recognition.face_encodings(rdj)[0]

elon_musk_image=face_recognition.load_image_file("photos/elon-musk.jpeg")
elon_musk_encoded=face_recognition.face_encodings(elon_musk_image)[0]

known_face_encoding=[
    rdj,
    elon_musk_encoded
]

known_face_names=[
    "rdj",
    "elon-musk"
]

students= known_face_names.copy()

face_loactions=[]
face_encodings=[]
face_names=[]
s=True

now=datetime.now()
current_date_time=now.strftime("%Y-%m-%d")

f=open(current_date_time+'.csv','w+',newline='')
lnwriter=csv.writer(f)

video_capture = cv2.VideoCapture(0)

while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]

    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_loactions)
        face_names=[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches(best_match_index):
                name= known_face_names[best_match_index]
            
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_date_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_date_time])

    cv2.imshow("attendance system sample", frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()




    
# why to use students arr is so we can avoid marking attendance for same student multiple times with
#students array what we do is after getting a face we see if its in db and in students arr if present
# we remove that name in students arr and mark the attendance for him this helps us in marking him
# attendance only once while taking multiple frames from video capture through camera