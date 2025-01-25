# Imports

import cv2 # OpenCV for video processing
import numpy # For mathematical operations
import time # For FPS calculation
from deepface import DeepFace # For emotion detecting (DeepFace is the dataset)
import tkinter as tk # For the GUI
from tkinter import ttk

# Turn on Camera
print("Turn on Camera?")
choice = input("Enter your choice (y or n): ")

if choice.lower() == 'y':
    cap = cv2.VideoCapture(0)  # Open Webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
elif choice.lower() == 'n':
    print("")
    exit()
else:
    print("Invalid choice. Exiting.")
    exit()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# List for previous emotions
emotion_history = []

# Process webcam feed
try:
    while True:
        start_time = time.time()  # Start time for FPS calculation
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Convert the frame to grayscale for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Initialise emotion as "No face detected"
        emotion = "No face detected"

        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a blue rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for Haar Cascade

            # Crop the face region
            face_roi = frame[y:y + h, x:x + w]

            # Perform DeepFace emotion detection
            try:
                analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # If `analysis` is a list, extract the first result
                if isinstance(analysis, list):
                    analysis = analysis[0]

                # Get the most dominant emotion
                emotion = analysis['dominant_emotion']

                # Add current detected emotion to list (only iif its different to the last one added)
                if len(emotion_history) == 0 or emotion_history[-1] != emotion:
                    emotion_history.append(emotion)

            except Exception as e:
                print("DeepFace error:", str(e))
                emotion = "Error detecting emotion"

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time + 1e-5)  # Small delta to avoid division by zero
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display detected emotion on video feed
        cv2.putText(frame, f"Emotion: {emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the video feed
        cv2.imshow("Webcam Feed", frame)

        # Create the main window
        window = create_widget(None, tk.Tk)
        window.title("GUI Example")

        # New frame for emotion history
        history_frame = numpy.zeros((300,400, 3), dtype=numpy.uint8)
        y_offset = 30
        for i, emo in enumerate(emotion_history):  
            text = f"{i + 1}: {emo}"
            cv2.putText(history_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        # Show emotion history frame
        cv2.imshow("Emotion History", history_frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed. Program exited.")