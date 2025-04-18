# Imports
import cv2 # OpenCV for video processing
import numpy as np # For mathematical operations
import time # For FPS calculation
from deepface import DeepFace # For emotion detecting (DeepFace is the dataset)
import tkinter as tk # For the GUI
from tkinter import ttk
import threading # For program windows to run concurrently
import mss # Multi-Screen Shot for screen processing
# win32 packages for screen processing
import win32gui
import win32con
import pygetwindow as gw
import win32api # To get screen resolution automatically
from configparser import ConfigParser # 
import os

'''
------------------------------------------------------------
Limited to Windows only, can't be used on mac or Linux
------------------------------------------------------------
'''

# Tkinter for GUI
root=tk.Tk()
root.title('EmotionLens ') # Window name
root.geometry("1000x600") # Size of the window


# Read config file and get colours

# Dynamically locate the config file in the same folder as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "setting_save.txt")

# Initialise config parser and settings
parser = ConfigParser()
saved_theme = 'Light'
saved_boundingbox_color = 'White'
saved_font_color = 'White'

# Read or create config file
try:
    if not parser.read(config_path):
        # Create default config
        parser['style'] = {'style': 'Light'}
        parser['bounding_box_color'] = {'bounding_box_color': 'White'}
        parser['font_color'] = {'font_color': 'White'}
        parser['camera'] = {'brightness': '50', 'contrast': '50'}

        with open(config_path, 'w') as configfile:
            parser.write(configfile)
    else:
        # Read existing config
        saved_theme = parser.get('style', 'style')
        saved_boundingbox_color = parser.get('bounding_box_color', 'bounding_box_color')
        saved_font_color = parser.get('font_color', 'font_color')
        brightness = int(parser.get('camera', 'brightness', fallback='50'))
        contrast = int(parser.get('camera', 'contrast', fallback='50'))
        
        # Apply saved settings immediately
        gui_style_mapping = {
            "Light": {"bg": "#f0f0f0", "fg": "black"},
            "Dark": {"bg": "#2e2e2e", "fg": "white"},
            "High Contrast": {"bg": "black", "fg": "yellow"},
        }

        color_mapping = {
            "Blue": (255, 0, 0),
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255),
            "Black": (0, 0, 0),
        }

        style = gui_style_mapping.get(saved_theme, {"bg": "#f0f0f0", "fg": "black"})
        bounding_box_color = color_mapping.get(saved_boundingbox_color, (255, 255, 255))
        font_color = color_mapping.get(saved_font_color, (255, 255, 255))

        root.configure(bg=style["bg"])
except Exception as e:
    print(f"Error reading config: {e}")

# Make capture a global variable
cap = None

# Function to get the screens resolution automatically
def get_screen_resolution():
    width = win32api.GetSystemMetrics(0)
    height = win32api.GetSystemMetrics(1)
    return width, height



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



# Function to detect faces on screen to start emotion detection
def start_screen_emotionDetection():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    emotion_history = []
    sct = mss.mss()
    monitor = sct.monitors[1]  # Primary monitor
    screen_width, screen_height = get_screen_resolution()

    # Create a transparent overlay window
    overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)  # Automatically adjusts screen size to current monitor
    cv2.namedWindow("EmotionLens Overlay", cv2.WINDOW_NORMAL) # Basic window
    cv2.setWindowProperty("EmotionLens Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # fULLSCREEN WINDOW FOR OVERLAY
    cv2.setWindowProperty("EmotionLens Overlay", cv2.WND_PROP_TOPMOST, 1) # Top-most layered window

    # Make the window transparent and click-through
    hwnd = win32gui.FindWindow(None, "EmotionLens Overlay")
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                          win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | # Preserve existing style
                          win32con.WS_EX_LAYERED | # Add transparency
                          win32con.WS_EX_TRANSPARENT) # Transparnet window ignores mouse clicks and doesnt interfere with other windows and/or screen
    win32gui.SetLayeredWindowAttributes(hwnd, 0, 0, win32con.LWA_COLORKEY) # Color key 0 = transparent

    # Process monitor feed
    try:
        while True:
            start_time = time.time() # Start time for FPS calculation
            screen = np.array(sct.grab(monitor)) # Monitor feed
            frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR) # Frame
            # Convert the frame to grayscale for Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # Clear overlay before drawing new elements
            overlay.fill(0)

            # Initialise emotion as "No face detected"
            emotion = "No face detected"

            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle with selected bounding box color
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0, 255), 2)

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
            cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, font_color, 2)

            # Display detected emotion on overlay
            cv2.putText(overlay, f"Emotion: {emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, font_color, 2)

            # Update overlay
            cv2.imshow("EmotionLens Overlay", overlay)

            # Exit loop when 'q' is pressed ONLY WORKS WHEN OVERLAY WINDOW IS MANUALLY SELECTED IN TASKBAR
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Cleanup resources
        cv2.destroyAllWindows()
        print("Monitor detection closed. Program exited.")



# Function to see settings for 'EmotionLens'
def settings_emotionLens():
    global bounding_box_color, font_color, brightness, contrast
    
    # Clear existing widgets in the window
    for widget in root.winfo_children():
        widget.destroy()

    # Create a settings frame
    settings_frame = tk.Frame(root)
    settings_frame.pack(pady=20)

    # Bounding Box Color Selection
    tk.Label(settings_frame, text="Change bounding box color:", font=("Helvetica", 12)).grid(row=0, column=0, padx=10, pady=5, sticky="w")
    boundingBox_color_combobox = ttk.Combobox(settings_frame, values=["White", "Black", "Red", "Green", "Blue", "Yellow"], state="readonly")
    boundingBox_color_combobox.grid(row=0, column=1, padx=10, pady=5)
    boundingBox_color_combobox.set(saved_boundingbox_color)

    # Font Color Selection
    tk.Label(settings_frame, text="Change font color:", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=5, sticky="w")
    fontColor_color_combobox = ttk.Combobox(settings_frame, values=["White", "Black", "Red", "Green", "Blue", "Yellow"], state="readonly")
    fontColor_color_combobox.grid(row=1, column=1, padx=10, pady=5)
    fontColor_color_combobox.set(saved_font_color)

    # GUI Theme Selection
    tk.Label(settings_frame, text="Select GUI Theme:", font=("Helvetica", 12)).grid(row=2, column=0, padx=10, pady=5, sticky="w")
    gui_theme_combobox = ttk.Combobox(settings_frame, values=["Light", "Dark", "High Contrast"], state="readonly")
    gui_theme_combobox.grid(row=2, column=1, padx=10, pady=5)
    gui_theme_combobox.set(saved_theme)

    # Resolution Placeholder
    tk.Label(settings_frame, text="Select monitor resolution (NOT WORKING):", font=("Helvetica", 12)).grid(row=3, column=0, padx=10, pady=5, sticky="w")
    monitor_resolution_combobox = ttk.Combobox(settings_frame, values=["1080 x 1920", "2560 x 1440", "3840 x 2160"], state="readonly")
    monitor_resolution_combobox.grid(row=3, column=1, padx=10, pady=5)
    monitor_resolution_combobox.set("1080 x 1920")

    # Save Settings Function
    def save_settings():
        color_mapping = {
            "Blue": (255, 0, 0),
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255),
            "Black": (0, 0, 0),
        }

        gui_style_mapping = {
            "Light": {"bg": "#f0f0f0", "fg": "black"},
            "Dark": {"bg": "#2e2e2e", "fg": "white"},
            "High Contrast": {"bg": "black", "fg": "yellow"},
        }

        # Get selected options from comboboxes
        selected_bb_color_name = boundingBox_color_combobox.get()
        selected_font_color_name = fontColor_color_combobox.get()
        selected_theme = gui_theme_combobox.get()

        # Update global values
        bounding_box_color = color_mapping[selected_bb_color_name]
        font_color = color_mapping[selected_font_color_name]
        style = gui_style_mapping[selected_theme]

        root.configure(bg=style["bg"])
        for widget in root.winfo_children():
            try:
                widget.configure(bg=style["bg"], fg=style["fg"])
            except:
                pass

        # Save settings to config file
        parser['bounding_box_color'] = {'bounding_box_color': selected_bb_color_name}
        parser['font_color'] = {'font_color': selected_font_color_name}
        parser['style'] = {'style': selected_theme}

        with open(config_path, 'w') as configfile:
            parser.write(configfile)

        print(f"Settings saved: Box Color={bounding_box_color}, Font Color={font_color}, Theme={selected_theme}")

    # Reset to Defaults Function
    def reset_to_defaults():
        boundingBox_color_combobox.set("White")
        fontColor_color_combobox.set("White")
        gui_theme_combobox.set("Light")

        parser['style'] = {'style': 'Light'}
        parser['bounding_box_color'] = {'bounding_box_color': 'White'}
        parser['font_color'] = {'font_color': 'White'}
        parser['camera'] = {'brightness': '50', 'contrast': '50'}

        with open(config_path, 'w') as configfile:
            parser.write(configfile)

        bounding_box_color = (255, 255, 255)
        font_color = (255, 255, 255)
        brightness = 50
        contrast = 50
        root.configure(bg="#f0f0f0")

        print("Settings reset to defaults.")

    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Save Settings", font=("Helvetica", 12), command=save_settings).pack(side="left", padx=10)
    tk.Button(button_frame, text="Calibrate Camera", font=("Helvetica", 12), command=calibrate_camera).pack(side="left", padx=10)
    tk.Button(button_frame, text="Reset to Defaults", font=("Helvetica", 12), command=reset_to_defaults).pack(side="left", padx=10)
    tk.Button(button_frame, text="Back", font=("Helvetica", 12), command=create_main_buttons).pack(side="left", padx=10)

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
    - Settings: Customise bounding box color, font size, and font color.
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
    cv2.setTrackbarPos("Brightness", "Calibration", brightness)
    cv2.setTrackbarPos("Contrast", "Calibration", contrast)

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

    parser['camera'] = {'brightness': str(brightness), 'contrast': str(contrast)}
    with open(config_path, 'w') as configfile:
        parser.write(configfile)



# Function to quit emotion detection 
def quit_emotionLens():
    global cap  # Access the global cap variable
    if cap is not None:  # If the camera is running, release it
        cap.release()
    cv2.destroyAllWindows()  # Close OpenCV windows
    root.quit()  # Exit the Tkinter mainloop
    root.destroy() # Complete termination



# Function to recreate the main buttons
def create_main_buttons():
    # Clear existing widgets in the window
    for widget in root.winfo_children():
        widget.destroy()

    title_label = tk.Label(root, text="EmotionLens - Emotion Detection System")
    title_label.pack(pady=20)

    start_button = tk.Button(root, text="Start Emotion Detection", command=start_emotionDetection)
    start_button.pack(pady=10)

    start_button = tk.Button(root, text="Detect Emotion From Screen", command=start_screen_emotionDetection)
    start_button.pack(pady=10)

    settings_button = tk.Button(root, text="Settings", command=settings_emotionLens)
    settings_button.pack(pady=10)

    helpGuide_button = tk.Button(root, text="Help Guide", command=helpGuide_emotionLens)
    helpGuide_button.pack(pady=10)

    quit_button = tk.Button(root, text="Quit", command=quit_emotionLens)
    quit_button.pack(pady=10)

# Initialise the main GUI with buttons
create_main_buttons()

# Start Tkinter window
root.mainloop()