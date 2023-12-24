import cv2
import numpy as np

# Load pre-trained models
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
vehicle_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Function to detect pedestrians in an image
def detect_pedestrians(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image

# Function to detect vehicles in an image
def detect_vehicles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in vehicles:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

# Capture video from the webcam (you can also use a video file)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect pedestrians and vehicles
    frame_with_pedestrians = detect_pedestrians(frame.copy())
    frame_with_vehicles = detect_vehicles(frame_with_pedestrians.copy())

    # Display the results
    cv2.imshow('Pedestrian and Vehicle Detection', frame_with_vehicles)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
