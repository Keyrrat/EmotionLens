# Imports
import cv2  # OpenCV for video processing
import numpy as np  # For mathematical operations
import time  # For FPS calculation
from deepface import DeepFace  # For emotion detection
import customtkinter as ctk  # For the modern GUI
from CTkMessagebox import CTkMessagebox
from tkinter import filedialog  # For file browsing
import threading  # For running detection in separate threads
import mss  # Multi-Screen Shot for screen capture
import win32gui, win32con, win32api  # For screen window management
import pygetwindow as gw  # Extra window controls
from configparser import ConfigParser  # For saving/loading settings
import os
import screeninfo  # For monitor info

'''
------------------------------------------------------------
Limited to Windows only, can't be used on Mac or Linux
------------------------------------------------------------
'''

# Initialise CustomTkinter for modern GUI appearance
ctk.set_appearance_mode("System")  # Default to system theme
ctk.set_default_color_theme("blue")  # Base theme

class EmotionLensApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title('EmotionLens')  # Window title
        self.geometry("1000x600")  # Initial window size
        self.minsize(800, 500)  # Minimum window size

        try:
            icon_path = os.path.join(os.path.dirname(__file__), "icon.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
            else:
                print("Error: icon.ico not found. Loading with no icon.")
        except Exception as e:
            print(f"Failed to load window icon: {e}")

        # Initialize variables with default values
        self.cap = None  # Video capture object
        self.bounding_box_colour = (255, 255, 255)  # BGR format for OpenCV
        self.font_colour = (255, 255, 255)  # Text colour for display
        self.brightness = 50  # Default camera brightness (0-100)
        self.contrast = 50  # Default camera contrast (0-100)
        self.available_monitors = []
        self.selected_monitor = 0
        self.available_cameras = []
        self.selected_camera = 0

        self.mode_var = ctk.StringVar()
        self.mode_display_var = ctk.StringVar()

        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setting_save.txt")
        self.parser = ConfigParser()
        self.load_config()

        saved_mode = self.parser.get('detection', 'mode', fallback='camera')
        self.mode_var.set(saved_mode)
        if saved_mode == "camera":
            self.mode_display_var.set("Camera")
        elif saved_mode == "screen":
            self.mode_display_var.set("Screen")
        elif saved_mode == "image":
            self.mode_display_var.set("Image")
        else:
            self.mode_display_var.set("Video")

        self.create_main_ui()

    # Load or create configuration file with saved settings
    def load_config(self):
        try:
            if not self.parser.read(self.config_path):
                # Create default config if file doesn't exist
                self.parser['style'] = {'style': 'System'}
                self.parser['bounding_box_colour'] = {'bounding_box_colour': 'White'}
                self.parser['font_colour'] = {'font_colour': 'White'}
                self.parser['camera'] = {'brightness': '50', 'contrast': '50', 'index': '0'}
                self.parser['monitor'] = {'index': '0'}
                self.parser['detection'] = {'mode': 'camera'}
                
                with open(self.config_path, 'w') as configfile:
                    self.parser.write(configfile)
            else:
                # Read existing config and apply settings
                saved_theme = self.parser.get('style', 'style', fallback='System')
                ctk.set_appearance_mode(saved_theme)
                
                self.bounding_box_colour = self.get_colour_from_name(self.parser.get('bounding_box_colour', 'bounding_box_colour', fallback='White'))
                self.font_colour = self.get_colour_from_name(self.parser.get('font_colour', 'font_colour', fallback='White'))
                self.brightness = int(self.parser.get('camera', 'brightness', fallback='50'))
                self.contrast = int(self.parser.get('camera', 'contrast', fallback='50'))
                self.selected_camera = int(self.parser.get('camera', 'index', fallback='0'))
                self.selected_monitor = int(self.parser.get('monitor', 'index', fallback='0'))
                self.mode_var.set(self.parser.get('detection', 'mode', fallback='camera'))

                self.detect_available_cameras()
                if self.selected_camera not in self.available_cameras:
                    self.selected_camera = self.available_cameras[0]

                self.available_monitors = screeninfo.get_monitors()
                if self.selected_monitor >= len(self.available_monitors):
                    self.selected_monitor = 0
        except Exception as e:
            print(f"Error reading config: {e}")

    def detect_available_cameras(self):
        self.available_cameras = []
        test_index = 0
        consecutive_failures = 0

        while consecutive_failures < 5:
            cap = cv2.VideoCapture(test_index)
            if cap.isOpened():
                self.available_cameras.append(test_index)
                cap.release()
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            test_index += 1

        if not self.available_cameras:
            self.available_cameras = [0]
            print("Warning: No cameras found, defaulting to index 0")

        self.selected_camera = self.available_cameras[0]

    def get_colour_from_name(self, colour_name):
        # OpenCV uses BGR format
        colour_mapping = {
            "Blue": (255, 0, 0),
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "White": (255, 255, 255),
            "Black": (0, 0, 0)
        }
        return colour_mapping.get(colour_name, (255, 255, 255))  # Default white

    # Create UI
    def create_main_ui(self):
        # Clear existing widgets if any
        for widget in self.winfo_children():
            widget.destroy()
        
        # Create tabbed interface
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Add tabs for different sections
        self.tabview.add("Main")  # Main detection controls
        self.tabview.add("Settings")  # Configuration options
        self.tabview.add("Help")  # User guide
        
        # Build each tab's content
        self.create_main_tab()
        self.create_settings_tab()
        self.create_help_tab()

    def create_main_tab(self):
        # Create content for the main detection tab
        tab = self.tabview.tab("Main")

        # Application title
        title = ctk.CTkLabel(tab, text="EmotionLens - Emotion Detection System", font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=20)

        # Detection mode selection (camera or screen)
        mode_frame = ctk.CTkFrame(tab, fg_color="transparent")
        mode_frame.pack(pady=10)

        self.mode_segment = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Camera", "Screen", "Image", "Video"],
            variable=self.mode_display_var,
            command=self.update_mode_selection,
            selected_color=("#3B8ED0", "#1F6AA5"),
            unselected_color=("gray75", "gray30"),
            selected_hover_color=("#36719F", "#1A5D8A"),
            font=ctk.CTkFont(weight="bold")
        )
        self.mode_segment.set(self.mode_display_var.get())
        self.mode_segment.pack(pady=5)

        # Main detection start button
        start_btn = ctk.CTkButton(
            tab,
            text="▶ Start Detection",
            command=self.start_detection,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2FA572",
            hover_color="#3DBA7C"
        )
        start_btn.pack(pady=20, ipadx=20, ipady=10)

    def update_mode_selection(self, selected_display):
        # Sync internal logic variable based on display
        mode_map = {"Camera": "camera", "Screen": "screen", "Image": "image", "Video": "video"}
        self.mode_var.set(mode_map.get(selected_display, "camera"))

        # Save new selection to config
        self.parser['detection'] = {'mode': self.mode_var.get()}
        with open(self.config_path, 'w') as configfile:
            self.parser.write(configfile)

    def create_settings_tab(self):
        # Create content for the settings configuration tab
        tab = self.tabview.tab("Settings")
        
        # Scrollable frame for settings options
        settings_frame = ctk.CTkScrollableFrame(tab)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Appearance settings section
        appearance_label = ctk.CTkLabel(settings_frame, text="Appearance", font=ctk.CTkFont(weight="bold"))
        appearance_label.pack(pady=(0, 10), anchor="w")
        
        # Theme selection dropdown
        theme_label = ctk.CTkLabel(settings_frame, text="Interface Theme:")
        theme_label.pack(anchor="w", pady=(5, 0))
        
        self.theme_var = ctk.StringVar(value=ctk.get_appearance_mode())
        theme_menu = ctk.CTkOptionMenu(settings_frame, values=["Light", "Dark", "System"], variable=self.theme_var, command=self.change_theme)
        theme_menu.pack(fill="x", pady=(0, 10))
        
        # Detection display settings section
        detection_label = ctk.CTkLabel(settings_frame, text="Detection Settings", font=ctk.CTkFont(weight="bold"))
        detection_label.pack(pady=(10, 10), anchor="w")
        
        # Bounding box colour selection
        bb_colour_label = ctk.CTkLabel(settings_frame, text="Bounding Box Colour:")
        bb_colour_label.pack(anchor="w", pady=(5, 0))
        self.bb_colour_var = ctk.StringVar(value=self.get_colour_name(self.bounding_box_colour))
        bb_colour_menu = ctk.CTkOptionMenu(settings_frame, values=["White", "Black", "Red", "Green", "Blue", "Yellow"], variable=self.bb_colour_var)
        bb_colour_menu.pack(fill="x", pady=(0, 10))
        
        # Text colour selection
        font_colour_label = ctk.CTkLabel(settings_frame, text="Text Colour:")
        font_colour_label.pack(anchor="w", pady=(5, 0))
        self.font_colour_var = ctk.StringVar(value=self.get_colour_name(self.font_colour))
        font_colour_menu = ctk.CTkOptionMenu(settings_frame, values=["White", "Black", "Red", "Green", "Blue", "Yellow"], variable=self.font_colour_var)
        font_colour_menu.pack(fill="x", pady=(0, 10))
        
        # Camera calibration button
        calibrate_btn = ctk.CTkButton(settings_frame, text="Calibrate Camera", command=self.calibrate_camera)
        calibrate_btn.pack(pady=10)
        
        # Settings action buttons
        save_btn = ctk.CTkButton(settings_frame, text="Save Settings", command=self.save_settings)
        save_btn.pack(pady=10)
        
        reset_btn = ctk.CTkButton(settings_frame, text="Reset to Default", command=self.reset_settings, fg_color="transparent", border_width=1)
        reset_btn.pack(pady=10)

    def create_help_tab(self):
        tab = self.tabview.tab("Help")
        
        # Help title
        help_title = ctk.CTkLabel(tab, text="Welcome to EmotionLens Help Guide!", font=ctk.CTkFont(size=22, weight="bold"))
        help_title.pack(pady=(30, 10), anchor="center")
        
        # Help text (no manual spaces needed)
        help_text = (
            "This application helps you detect emotions in real-time using your webcam.\n\n"
            "- Start Emotion Detection: Begin detecting emotions in real time.\n"
            "- Settings: Customize bounding box color, font size, and font color.\n"
            "- Calibrate Camera: Calibrate your camera for better accuracy.\n"
            "- Quit: Exit the application.\n\n"
            "Press 'q' during detection to stop the webcam or screen feed."
        )
        
        help_label = ctk.CTkLabel(tab, text=help_text, justify="center", font=ctk.CTkFont(size=16), wraplength=700)
        help_label.pack(padx=20, pady=10, anchor="center")

    def get_colour_name(self, colour):
        # Convert BGR tuple to colour name
        colour_mapping = {
            (255, 0, 0): "Blue",
            (0, 0, 255): "Red",
            (0, 255, 0): "Green",
            (0, 255, 255): "Yellow",
            (255, 255, 255): "White",
            (0, 0, 0): "Black",
        }
        return colour_mapping.get(tuple(colour), "White")
    
    def change_theme(self, new_theme):
        # Change the application theme dynamically
        ctk.set_appearance_mode(new_theme)
    
    def save_settings(self):
        # Save current settings to config file
        try:
            # Update runtime settings from UI selections
            self.bounding_box_colour = self.get_colour_from_name(self.bb_colour_var.get())
            self.font_colour = self.get_colour_from_name(self.font_colour_var.get())
            
            # Save all settings to config file
            self.parser['style'] = {'style': self.theme_var.get()}
            self.parser['bounding_box_colour'] = {'bounding_box_colour': self.bb_colour_var.get()}
            self.parser['font_colour'] = {'font_colour': self.font_colour_var.get()}
            self.parser['camera'] = {'brightness': str(self.brightness), 'contrast': str(self.contrast), 'index': str(self.selected_camera)}
            self.parser['monitor'] = {'index': str(self.selected_monitor)}
            self.parser['detection'] = {'mode': self.mode_var.get()}
            
            with open(self.config_path, 'w') as configfile:
                self.parser.write(configfile)
            
            print("Settings saved successfully")
            CTkMessagebox(title="Success", message="Settings saved successfully!", icon="check")
        except Exception as e:
            print(f"Error saving settings: {e}")
            CTkMessagebox(title="Error", message="Error saving settings!", icon="cancel")
    
    def reset_settings(self):
        # Reset all settings to default values
        self.theme_var.set("System")
        self.bb_colour_var.set("White")
        self.font_colour_var.set("White")
        self.brightness = 50
        self.contrast = 50
        self.mode_var.set("camera")
        
        # Update runtime settings
        self.bounding_box_colour = self.get_colour_from_name("White")
        self.font_colour = self.get_colour_from_name("White")
        
        # Save defaults to config file
        self.save_settings()

        print("Settings reset successfully")
        CTkMessagebox(title="Reset", message="Settings reset to default.", icon="warning")
    
    def start_detection(self):
        # Start emotion detection in a separate thread based on selected mode
        mode = self.mode_var.get()
        if mode == "camera":
            threading.Thread(target=self.start_emotionDetection, daemon=True).start()
        elif mode == "screen":
            threading.Thread(target=self.start_screen_emotionDetection, daemon=True).start()
        elif mode == "image":
            threading.Thread(target=self.image_emotionDetection, daemon=True).start()
        elif mode == "video":
            threading.Thread(target=self.video_emotionDetection, daemon=True).start()
        else:
            print("Error: Unknown mode selected")

    def image_emotionDetection(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            print("Error: No file selected")
            return
        
        frame = cv2.imread(file_path) # Load the frame
        if frame is None:
            print(f"Error: Cannot load image at {file_path}")
            return

        # Minimum dimensions
        min_width = 500
        min_height = 500

        height, width = frame.shape[:2]
        if width < min_width or height < min_height:
            scale_w = min_width / width
            scale_h = min_height / height
            scale = max(scale_w, scale_h)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"Image resized to: {new_width}x{new_height}")

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
            print("Dominant emotion:", dominant_emotion)

            cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Image Emotion Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"DeepFace error: {e}")
    
    def video_emotionDetection(self):
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")]
        )
        if not video_path:
            print("Error: No file selected")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        frame_count = 0
        last_emotion = "Detecting..."

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video or failed reading frame.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.bounding_box_colour, 2)

                    # Only run DeepFace every 5 frames
                    if frame_count % 5 == 0:
                        face_roi = frame[y:y+h, x:x+w]
                        try:
                            analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            last_emotion = analysis["dominant_emotion"]
                        except Exception as e:
                            print("DeepFace error:", str(e))
                            last_emotion = "Error"

                    text_position = (x, y + h + 30)
                    cv2.putText(frame, f"{last_emotion}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.font_colour, 2)

                cv2.imshow("Video Emotion Detection", frame)

                frame_count += 1

                if cv2.waitKey(30) & 0xFF == 27:  # ~30ms delay (approx 30 FPS)
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def start_emotionDetection(self):
        # Perform real-time emotion detection from webcam feed
        self.cap = cv2.VideoCapture(self.selected_camera)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.selected_camera}.")
            msg = CTkMessagebox(
                title="Error", 
                message="Could not open webcam.", 
                icon="cancel", 
                option_1="Retry", 
                option_2="Cancel"
            )
            if msg.get() == "Retry":
                self.start_emotionDetection()
            return
        
        # Apply saved camera settings
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness / 100)
        self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast / 100)
        
        # Load face detection classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        emotion_history = []  # Track emotion changes over time
        
        try:
            while True:
                start_time = time.time()  # For FPS calculation
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture image")
                    break
                
                frame = cv2.flip(frame, 1)  # Mirror the image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                
                # Detect faces using Haar Cascade
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                emotion = "No face detected"  # Default message
                
                for (x, y, w, h) in faces:
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.bounding_box_colour, 2)
                    face_roi = frame[y:y + h, x:x + w]
                    
                    try:
                        analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        emotion = analysis["dominant_emotion"]
                    except Exception as e:
                        print("DeepFace error:", str(e))
                        emotion = "Error"

                    text_position = (x, y + h + 30)
                    cv2.putText(frame, f"{emotion}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.font_colour, 2)

                
                # Show the video feed
                cv2.imshow("Webcam Feed", frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            # Cleanup resources
            self.cap.release()
            cv2.destroyAllWindows()

    def start_screen_emotionDetection(self):
        # Detect emotion from screen using transparent overlay
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        emotion_history = []
        sct = mss.mss()  # Screen capture tool
        monitor = sct.monitors[self.selected_monitor + 1]  # Selected monitor
        
        # Get screen resolution
        screen_width = monitor['width']
        screen_height = monitor['height']
        screen_left = monitor['left']
        screen_top = monitor['top']
        
        # Create transparent overlay window
        overlay = np.zeros((screen_height, screen_width, 4), dtype=np.uint8)
        cv2.namedWindow("EmotionLens Overlay", cv2.WINDOW_NORMAL)
        cv2.moveWindow("EmotionLens Overlay", screen_left, screen_top)
        cv2.setWindowProperty("EmotionLens Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty("EmotionLens Overlay", cv2.WND_PROP_TOPMOST, 1)
        
        # Configure window transparency properties
        hwnd = win32gui.FindWindow(None, "EmotionLens Overlay")
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
        win32gui.SetLayeredWindowAttributes(hwnd, 0, 0, win32con.LWA_COLORKEY)
        
        try:
            while True:
                start_time = time.time()
                screen = np.array(sct.grab(monitor))  # Capture screen
                frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                overlay.fill(0)  # Clear previous frame
                emotion = "No face detected"
                
                for (x, y, w, h) in faces:
                    # Draw box
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (*self.bounding_box_colour, 255), 2)
                    face_roi = frame[y:y + h, x:x + w]
                    
                    try:
                        analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        emotion = analysis["dominant_emotion"]
                        
                        # Draw emotion BELOW the bounding box
                        text_position = (x, y + h + 30)
                        cv2.putText(overlay, f"{emotion}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (*self.font_colour, 255), 2)
                    except Exception as e:
                        print("DeepFace error:", str(e))
                
                # Display metrics on overlay
                fps = 1 / (time.time() - start_time + 1e-5)
                cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (*self.font_colour, 255), 2)
                
                cv2.imshow("EmotionLens Overlay", overlay)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cv2.destroyAllWindows()
    
    def calibrate_camera(self):
        # Adjust camera brightness and contrast settings
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.selected_camera)
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                msg = CTkMessagebox(
                    title="Error", 
                    message="Could not open webcam for calibration.", 
                    icon="cancel", 
                    option_1="Retry", 
                    option_2="Cancel"
                )
                if msg.get() == "Retry":
                    self.calibrate_camera()
                return

        cv2.namedWindow("Calibration")
        cv2.createTrackbar("Brightness", "Calibration", self.brightness, 100, lambda x: None)
        cv2.createTrackbar("Contrast", "Calibration", self.contrast, 100, lambda x: None)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            frame = cv2.flip(frame, 1)
            brightness = cv2.getTrackbarPos("Brightness", "Calibration")
            contrast = cv2.getTrackbarPos("Contrast", "Calibration")
            
            # Apply adjustments
            alpha = contrast / 50 + 0.5
            beta = brightness - 50
            adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
            
            cv2.imshow("Calibration", adjusted_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                self.brightness = brightness
                self.contrast = contrast
                break
        
        cv2.destroyWindow("Calibration")
        print(f"Camera calibration saved. Brightness: {self.brightness}, Contrast: {self.contrast}")
        
        # Save camera settings
        self.parser['camera'] = {'brightness': str(self.brightness), 'contrast': str(self.contrast), 'index': str(self.selected_camera)}
        with open(self.config_path, 'w') as configfile:
            self.parser.write(configfile)

# Run application
if __name__ == "__main__":
    app = EmotionLensApp()
    app.mainloop()
