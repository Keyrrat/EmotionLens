# Imports

import cv2 # OpenCV for video processing
import numpy # For mathematical operations
import time # Time module
import os

# Turn on Camera
print("Turn on Camera?")
choice = input("Enter your choice (y or n): ")

if choice == 'y':
    # Open Camera
    cap = cv2.VideoCapture(0) # Open Webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    mode = 'webcam'
elif choice == 'n':
    print("")
    exit()
else:
    print("Invalid choice. Exiting.")
    exit()

# Process webcam feed
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    height, width, _ = frame.shape  # Get frame dimensions

    # Calculate and display FPS
    fps = 1 / (time.time() - start_time + 1e-5)  # Add a small delta to avoid division by zero
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # Displays FPS

    # Display the video feed
    cv2.imshow("Webcam Feed", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()
print("Webcam closed. Program exited.")