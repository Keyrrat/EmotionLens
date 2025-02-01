# Imports

import cv2 # OpenCV for video processing
import numpy # For mathematical operations
import time # For FPS calculation
from deepface import DeepFace # For emotion detecting (DeepFace is the dataset)
import tkinter as tk # For the GUI
from tkinter import ttk

# Tkinter for GUI
root=tk.Tk()
root.title('EmotionLens - ')
root.geometry("800x600") # Size of the window

# Make capture a global variable
cap = None

# Calibration variables
brightness = 50 #defaukt brightness
contrast = 50 #Default contrast

# Global settings (defaults)
bounding_box_color = (255, 0, 0)  # Default Blue
font_color = (255, 255, 255)  # Default White

# Function to start emotion detection when the user clicks the start button on the GUI
def start_emotionDetection():

# Turn on Camera
    global cap, bounding_box_color, font_color  # Access the global cap variable
    cap = cv2.VideoCapture(0)  # Open Webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
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
                return

            # Mirror the frame
            frame = cv2.flip(frame, 1)  # Flip horizontally

            # Convert the frame to grayscale for Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Initialise emotion as "No face detected"
            emotion = "No face detected"

            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle with selected bounding box color
                cv2.rectangle(frame, (x, y), (x + w, y + h), bounding_box_color, 2)

                # Crop the face region
                face_roi = frame[y:y + h, x:x + w]

                # Perform DeepFace emotion detection
                try:
                    analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)

                    # If `analysis` is a list, extract the first result
                    if isinstance(analysis, list):
                        analysis = analysis[0]

                    # Get the most dominant emotion
                    emotion = analysis["dominant_emotion"]

                    # Add current detected emotion to list (only if different from last one)
                    if len(emotion_history) == 0 or emotion_history[-1] != emotion:
                        emotion_history.append(emotion)

                except Exception as e:
                    print("DeepFace error:", str(e))
                    emotion = "Error detecting emotion"

            # Calculate and display FPS
            fps = 1 / (time.time() - start_time + 1e-5)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, font_color, 2)

            # Display detected emotion on video feed
            cv2.putText(frame, f"Emotion: {emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, font_color, 2)

            # Show the video feed
            cv2.imshow("Webcam Feed", frame)

            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Cleanup resources
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed. Program exited.")



# Function to see settings for 'EmotionLens'
def settings_emotionLens():
    global bounding_box_color, font_color

    # Clear existing widgets in the window
    for widget in root.winfo_children():
        widget.destroy()

    # Create a settings frame
    settings_frame = tk.Frame(root)
    settings_frame.pack(pady=20)

    # Bounding Box Color Selection
    boundingBox_label = tk.Label(settings_frame, text="Change bounding box color:", font=("Helvetica", 12))
    boundingBox_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    boundingBox_color_combobox = ttk.Combobox(
        settings_frame, values=["Blue", "Red", "Green", "Yellow"], state="readonly"
    )
    boundingBox_color_combobox.grid(row=0, column=1, padx=10, pady=5)
    boundingBox_color_combobox.set("Blue")  # Default selection

    # Font Color Selection
    fontColor_label = tk.Label(settings_frame, text="Change font color:", font=("Helvetica", 12))
    fontColor_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

    fontColor_color_combobox = ttk.Combobox(
        settings_frame, values=["White", "Black", "Red", "Green"], state="readonly"
    )
    fontColor_color_combobox.grid(row=1, column=1, padx=10, pady=5)
    fontColor_color_combobox.set("White")  # Default selection


 # Save Settings Function
    def save_settings():
        global bounding_box_color, font_color

        # Map colors to OpenCV BGR format
        color_mapping = {
            "Blue": (255, 0, 0),
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255),
            "Black": (0, 0, 0),
        }

        # Save the chosen colors
        bounding_box_color = color_mapping[boundingBox_color_combobox.get()]
        font_color = color_mapping[fontColor_color_combobox.get()]

        print(f"Settings saved: Bounding Box Color: {bounding_box_color}, Font Color: {font_color}")

    # Save Button
    save_button = tk.Button(settings_frame, text="Save Settings", font=("Helvetica", 12), command=save_settings)
    save_button.grid(row=2, column=0, columnspan=2, pady=20)

    # Back Button
    back_button = tk.Button(settings_frame, text="Back", font=("Helvetica", 12), command=create_main_buttons)
    back_button.grid(row=3, column=0, columnspan=2, pady=20)


# Function to see help guide
def helpGuide_emotionLens():
    # Clear existing widgets in the window
    for widget in root.winfo_children():
        widget.destroy()

    # Create a help guide frame (expanded to fill the window properly)
    helpGuide_frame = tk.Frame(root)
    helpGuide_frame.pack(expand=True, fill="both")

    # Help guide info
    helpGuide_label = tk.Label(
        helpGuide_frame,
        text="""
    Welcome to EmotionLens Help Guide!
    This application helps you detect emotions in real-time using your webcam.

    - Start Emotion Detection: Begin detecting emotions in real time.
    - Settings: Customize bounding box color, font size, and font color.
    - Calibrate Camera: Calibrate your camera for better accuracy.
    - Quit: Exit the application.

    Press 'q' during emotion detection to stop the webcam.
    """,
        font=("Helvetica", 12), justify="left", anchor="w",
    )
    helpGuide_label.pack(pady=20, padx=20, anchor="w")

    # Back button
    back_button = tk.Button(helpGuide_frame, text="Back", font=("Helvetica", 12), command=create_main_buttons)
    back_button.pack(pady=20)



# Function to calibrate camera
def calibrate_camera():
    global cap, brightness, contrast

    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Open Webcam
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

    # Create a window for calibration settings
    cv2.namedWindow("Calibration")

    # Callback function for trackbars (does nothing but required)
    def nothing(x):
        pass

    # Create trackbars for brightness and contrast
    cv2.createTrackbar("Brightness", "Calibration", brightness, 100, nothing)
    cv2.createTrackbar("Contrast", "Calibration", contrast, 100, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Get values from trackbars
        brightness = cv2.getTrackbarPos("Brightness", "Calibration")
        contrast = cv2.getTrackbarPos("Contrast", "Calibration")

        # Apply brightness and contrast adjustments
        alpha = contrast / 50 + 0.5  # Scale contrast
        beta = brightness - 50  # Adjust brightness
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # Show the adjusted video feed
        cv2.imshow("Calibration", adjusted_frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close calibration window but keep the camera open
    cv2.destroyWindow("Calibration")
    print(f"Camera calibration saved. Brightness: {brightness}, Contrast: {contrast}")


# Function to quit emotion detection 
def quit_emotionLens():
    global cap  # Access the global cap variable
    if cap is not None:  # If the camera is running, release it
        cap.release()
    cv2.destroyAllWindows()  # Close OpenCV windows
    root.quit()  # Exit the Tkinter mainloop




# Function to recreate the main buttons
def create_main_buttons():
    # Clear existing widgets in the window
    for widget in root.winfo_children():
        widget.destroy()

    title_label = tk.Label(root, text="EmotionLens - Emotion Detection System")
    title_label.pack(pady=20)

    start_button = tk.Button(root, text="Start Emotion Detection", command=start_emotionDetection)
    start_button.pack(pady=10)

    settings_button = tk.Button(root, text="Settings", command=settings_emotionLens)
    settings_button.pack(pady=10)

    helpGuide_button = tk.Button(root, text="Help Guide", command=helpGuide_emotionLens)
    helpGuide_button.pack(pady=10)

    calibrateCamera_button = tk.Button(root, text="Calibrate Camera", command=calibrate_camera)
    calibrateCamera_button.pack(pady=10)

    quit_button = tk.Button(root, text="Quit", command=quit_emotionLens)
    quit_button.pack(pady=10)

# Initialise the main GUI with buttons
create_main_buttons()

# Start Tkinter window
root.mainloop()