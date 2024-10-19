import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file("ImageBasic/elon test.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElonTest = face_recognition.load_image_file("ImageBasic/elon verify.jpg")
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

imgBillTest = face_recognition.load_image_file("ImageBasic/Bill gates.jpg")
imgBillTest = cv2.cvtColor(imgBillTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)
encodings = face_recognition.face_encodings(imgElon, faceLoc)
cv2.rectangle(imgElon, (faceLoc[0][3], faceLoc[0][0]), (faceLoc[0][1], faceLoc[0][2]),(255,0,255), 2)

faceLocTest = face_recognition.face_locations(imgElonTest)
encodingsTest = face_recognition.face_encodings(imgElonTest, faceLocTest)
cv2.rectangle(imgElonTest, (faceLocTest[0][3], faceLocTest[0][0]), (faceLocTest[0][1], faceLocTest[0][2]),(255,0,255), 2)

results = face_recognition.compare_faces(np.array(encodings), encodingsTest)
faceDis = face_recognition.face_distance(np.array(encodings), encodingsTest)
print(results, faceDis)
cv2.putText(imgElonTest, f'{results} {round(faceDis[0], 2)}', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.imshow("Elon", imgElon)
cv2.imshow("ElonTest", imgElonTest)
cv2.waitKey(0)