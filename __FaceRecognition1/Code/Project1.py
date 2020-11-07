import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('code/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('code/haarcascade_eye.xml')


# Read the input image
img = cv2.imread('../images/Joanna&I.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces


faces = face_cascade.detectMultiScale(gray, 1.1, 4)
eyes = eye_cascade.detectMultiScale(gray,1.1,4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
  
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
# Display the output
cv2.imshow('img', img)



cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)