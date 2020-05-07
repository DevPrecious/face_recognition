import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


img = cv2.imread('nn.jpg')
faces = face_cascade.detectMultiScale(img, 1.3, 5)

print(f'faces found {len(faces)}')
print(img.shape)
print('Cord', faces)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    roi_face = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_face, (ex,ey),(ex+ew, ey+eh), (255,0,0), 2)

font = cv2.FONT_HERSHEY_SIMPLEX
text = cv2.putText(img, 'Face detected', (0, img.shape[0]), font, 2, (255,255,255), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()