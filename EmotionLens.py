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
            icon_path = os.path.join(os.path.dirname(__file__), "icon_emotionLens.ico")
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
        
        # Emotion history tracking
        self.emotion_history = []
        self.last_emotion = None
        self.last_update_time = 0
        self.history_panel_height = 300
        self.animation_step = 0
        self.animation_running = False
        self.history_panel_visible = False

        self.mode_var = ctk.StringVar()
        self.mode_display_var = ctk.StringVar()
        self.detection_running = False  # Track if detection is active
        self.detection_thread = None  # Save the detection thread

        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setting_save.txt")
        self.parser = ConfigParser()
        self.load_config()

        saved_mode = self.parser.get('detection', 'mode', fallback='camera')
        self.mode_var.set(saved_mode)
        self.mode_display_var.set("Camera" if saved_mode == "camera" else 
                                 "Screen" if saved_mode == "screen" else
                                 "Image" if saved_mode == "image" else "Video")

        self.create_main_ui()

    # Load or create configuration file with saved settings
    def load_config(self):
        try:
            if not self.parser.read(self.config_path):
                # Create default config if file doesn't exist
                self.parser['style'] = {'style': 'System'}
                self.parser['bounding_box_colour'] = {'bounding_box_colour': 'White'}
                self.parser['font_colour'] = {'font_colour': 'White'}
                self.parser['text_size'] = {'text_size': 'Medium'}
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
                self.text_size = self.get_text_size(self.parser.get('text_size', 'text_size', fallback='Medium'))
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
        self.mode_segment.pack(pady=5)

        # Main detection start button
        self.start_stop_btn = ctk.CTkButton(
            tab,
            text="▶ Start Detection",
            command=self.toggle_detection,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2FA572",
            hover_color="#3DBA7C"
        )
        self.start_stop_btn.pack(pady=20, ipadx=20, ipady=10)

        self.history_btn = ctk.CTkButton(
            tab,
            text="🕘 View Emotion History",
            command=self.show_history_panel,
            fg_color="#3B8ED0",
            hover_color="#36719F"
        )
        self.history_btn.pack(side="bottom", pady=10)

        self.create_history_panel()

    def create_history_panel(self):
        self.history_container = ctk.CTkFrame(
            self.tabview.tab("Main"),
            fg_color="transparent",
            width=self.winfo_width(),
            height=0
        )
        self.history_container.pack_propagate(False)
        self.history_container.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)

        self.history_panel = ctk.CTkFrame(
            self.history_container,
            fg_color=("#f0f0f0", "#2b2b2b"),
            height=self.history_panel_height
        )
        self.history_panel.pack(fill="both", expand=True)

        title_frame = ctk.CTkFrame(self.history_panel, fg_color="transparent")
        title_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(title_frame, text="Emotion History", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        
        close_btn = ctk.CTkButton(
            title_frame,
            text="×",
            width=30,
            height=30,
            command=self.hide_history_panel,
            fg_color="transparent",
            hover_color=("#e0e0e0", "#333333"),
            font=ctk.CTkFont(size=16, weight="bold")
        )
        close_btn.pack(side="right")
        
        self.history_scroll = ctk.CTkScrollableFrame(
            self.history_panel,
            fg_color="transparent"
        )
        self.history_scroll.pack(fill="both", expand=True, padx=10, pady=(0,10))

    def update_emotion_history(self, emotion):
        if not self.detection_running:
            return
            
        current_time = time.time()
        if emotion != self.last_emotion and (current_time - self.last_update_time) >= 2:
            self.last_emotion = emotion
            self.last_update_time = current_time
            timestamp = time.strftime("%H:%M:%S")
            self.emotion_history.append(f"{timestamp}  =  {emotion}")
            
            if len(self.emotion_history) > 20:
                self.emotion_history = self.emotion_history[-20:]
            
            if hasattr(self, 'history_panel_visible') and self.history_panel_visible:
                self.update_history_content()

    def update_history_content(self):
        if hasattr(self, 'history_scroll'):
            for widget in self.history_scroll.winfo_children():
                widget.destroy()
            
            if not self.emotion_history:
                ctk.CTkLabel(
                    self.history_scroll,
                    text="No emotion history available\nStart detection to record emotions",
                    justify="center"
                ).pack(pady=20)
            else:
                for item in reversed(self.emotion_history):
                    entry_frame = ctk.CTkFrame(self.history_scroll, fg_color="transparent")
                    entry_frame.pack(fill="x", pady=2)
                    
                    timestamp, emotion = item.split("  =  ")
                    
                    ctk.CTkLabel(
                        entry_frame,
                        text=timestamp,
                        width=100,
                        anchor="w",
                        font=ctk.CTkFont(weight="bold")
                    ).pack(side="left", padx=(0,10))
                    
                    ctk.CTkLabel(
                        entry_frame,
                        text=emotion,
                        anchor="w"
                    ).pack(side="left", fill="x", expand=True)

    def start_detection(self):
        if not self.detection_running:
            self.emotion_history = []
            self.last_emotion = None
            self.last_update_time = 0
            
        self.detection_running = True
        self.update_start_stop_button()
        
        mode = self.mode_var.get()
        if mode == "camera":
            self.detection_thread = threading.Thread(target=self.start_emotionDetection, daemon=True)
        elif mode == "screen":
            self.detection_thread = threading.Thread(target=self.start_screen_emotionDetection, daemon=True)
        elif mode == "image":
            self.detection_thread = threading.Thread(target=self.image_emotionDetection, daemon=True)
        elif mode == "video":
            self.detection_thread = threading.Thread(target=self.video_emotionDetection, daemon=True)
        
        self.detection_thread.start()

    def stop_detection(self):
        self.detection_running = False
        self.update_start_stop_button()
        if hasattr(self, 'history_panel_visible') and self.history_panel_visible:
            self.update_history_content()

    def show_history_panel(self):
        if not hasattr(self, 'history_panel'):
            self.create_history_panel()
        
        if not self.history_panel_visible:
            self.history_panel_visible = True
            self.animate_panel(show=True)
            self.update_history_content()

    def hide_history_panel(self):
        if hasattr(self, 'history_panel') and self.history_panel_visible:
            self.history_panel_visible = False
            self.animate_panel(show=False)

    def animate_panel(self, show=True):
        if self.animation_running:
            return
            
        self.animation_running = True
        
        def update():
            if show:
                self.animation_step += 15
                if self.animation_step >= self.history_panel_height:
                    self.animation_step = self.history_panel_height
                    self.animation_running = False
                    return
            else:
                self.animation_step -= 15
                if self.animation_step <= 0:
                    self.animation_step = 0
                    self.animation_running = False
                    return
                    
            self.history_container.configure(height=self.animation_step)
            self.after(10, update)
        
        update()

    def update_mode_selection(self, selected_display):
        # Sync internal logic variable based on display
        mode_map = {"Camera": "camera", "Screen": "screen", "Image": "image", "Video": "video"}
        self.mode_var.set(mode_map.get(selected_display, "camera"))

        # Save new selection to config
        self.parser['detection'] = {'mode': self.mode_var.get()}
        with open(self.config_path, 'w') as configfile:
            self.parser.write(configfile)
            
    def toggle_detection(self):
        if not self.detection_running:
            self.start_detection() # Start detection
        else:
            self.stop_detection() # Stop detection

    def update_start_stop_button(self):
        if self.detection_running:
            self.start_stop_btn.configure(
                text="■ Stop Detection",
                fg_color="#D0312D",
                hover_color="#AD1D1D"
            )
        else:
            self.start_stop_btn.configure(
                text="▶ Start Detection",
                fg_color="#2FA572",
                hover_color="#3DBA7C"
            )

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
        detection_label = ctk.CTkLabel(settings_frame, text="⚙️ Detection Settings", font=ctk.CTkFont(weight="bold"))
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

        # Text size selection
        text_size_label = ctk.CTkLabel(settings_frame, text="Text Size:")
        text_size_label.pack(anchor="w", pady=(5, 0))
        self.text_size_var = ctk.StringVar(value="Medium")  # Default to Medium
        text_size_menu = ctk.CTkOptionMenu(settings_frame, values=["Small", "Medium", "Large", "Extra Large"], variable=self.text_size_var)
        text_size_menu.pack(fill="x", pady=(0, 10))
        
        # Camera calibration button
        calibrate_btn = ctk.CTkButton(settings_frame, text="📽 Calibrate Camera", command=self.calibrate_camera)
        calibrate_btn.pack(pady=10)
        
        # Settings action buttons
        save_btn = ctk.CTkButton(settings_frame, text="💾 Save Settings", command=self.save_settings)
        save_btn.pack(pady=10)
        
        reset_btn = ctk.CTkButton(settings_frame, text="Reset to Default", command=self.reset_settings, fg_color="transparent", border_width=1)
        reset_btn.pack(pady=10)

    def create_help_tab(self):
        tab = self.tabview.tab("Help")

        # Scrollable frame for long help text
        help_frame = ctk.CTkScrollableFrame(tab)
        help_frame.pack(fill="both", expand=True, padx=20, pady=20)

        help_title = ctk.CTkLabel(
            help_frame,
            text="Welcome to EmotionLens Help Guide!",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        help_title.pack(pady=(10, 10))

        # Inner frame to centre content
        inner_frame = ctk.CTkFrame(help_frame, fg_color="transparent")
        inner_frame.pack(anchor="center")

        help_text = (
            "Welcome To EmotionLens - Guide \n\n"
            "📷 Detection Modes:\n"
            " - Camera: Detects emotions in real-time from your webcam.\n"
            " - Screen: Detects emotions from people/ faces visible on your screen via a transparent overlay.\n"
            " - Image: Select an image to detect visible faces and their emotions.\n"
            " - Video: Load a video file and detect emotions for each face.\n\n"
            "⚙️ Settings:\n"
            " - Bounding Box Colour: You can change the colour of rectangles around detected faces.\n"
            " - Text Colour: Users can change the colour of the emotion labels.\n"
            " - Text Size: USers can adjust how large the emotion labels are.\n"
            " - Theme: Users can choose between Light, Dark, or System theme.\n\n"
            "📽 Camera Calibration:\n"
            " - Open a live preview to adjust Brightness and Contrast using sliders.\n"
            " - Press [Esc] to save your changes and return to the app.\n\n"
            "💾 Saving Settings:\n"
            " - Use 'Save Settings' to save your custom settings.\n"
            " - Use 'Reset to Default' to reset all options.\n\n"
            "🛑 Stopping Detection:\n"
            " - Press the [Esc] key or click '■ Stop Detection' to exit any detection mode.\n"
            " - Closing the detection window will also stop detection.\n\n"
            "❓ Having trouble?\n"
            " - Make sure your camera is not being used by any other app.\n"
            " - For screen detection, make sure the overlay window stays open.\n"
            " - Use the calibration tool to help improve detection accuracy.\n\n"
            "The Screen, Image, and Video options are currently available only on Windows."
        )

        help_label = ctk.CTkLabel(
        inner_frame,
        text=help_text,
        justify="left",
        font=ctk.CTkFont(size=16),
        wraplength=800
        )
        help_label.pack()

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
    
    def get_text_size(self, size_name):
        size_mapping = {
            "Small": 0.5,
            "Medium": 0.7,
            "Large": 1.0,
            "Extra Large": 1.2
        }
        return size_mapping.get(size_name, 0.7)  # Default to Medium size
    
    def change_theme(self, new_theme):
        # Change the application theme dynamically
        ctk.set_appearance_mode(new_theme)
    
    def save_settings(self):
        # Save current settings to config file
        try:
            # Update runtime settings from UI selections
            self.bounding_box_colour = self.get_colour_from_name(self.bb_colour_var.get())
            self.font_colour = self.get_colour_from_name(self.font_colour_var.get())
            self.text_size = self.get_text_size(self.text_size_var.get())
            
            # Save all settings to config file
            self.parser['style'] = {'style': self.theme_var.get()}
            self.parser['bounding_box_colour'] = {'bounding_box_colour': self.bb_colour_var.get()}
            self.parser['font_colour'] = {'font_colour': self.font_colour_var.get()}
            self.parser['text_size'] = {'text_size': self.text_size_var.get()}
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
        self.text_size_var.set("Medium")
        self.brightness = 50
        self.contrast = 50
        self.mode_var.set("camera")
        
        # Update runtime settings
        self.bounding_box_colour = self.get_colour_from_name("White")
        self.font_colour = self.get_colour_from_name("White")
        self.text_size = self.get_text_size("Medium")
        
        # Save defaults to config file
        self.save_settings()

        print("Settings reset successfully")
        CTkMessagebox(title="Reset", message="Settings reset to default.", icon="warning")

    def image_emotionDetection(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            print("Error: No file selected")
            self.stop_detection()
            return

        frame = cv2.imread(file_path)
        if frame is None:
            print(f"Error: Cannot load image at {file_path}")
            self.stop_detection()
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
            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                # If no face detected, analyse the whole frame
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
                print("Dominant emotion (whole frame):", dominant_emotion)
                cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # If faces detected, analyse each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.bounding_box_colour, 2)
                    face_roi = frame[y:y+h, x:x+w]
                    try:
                        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                        dominant_emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis['dominant_emotion']
                        print("Dominant emotion (face):", dominant_emotion)
                    except Exception as e:
                        print(f"DeepFace error on face: {e}")
                        dominant_emotion = "Error"

                    text_position = (x, y + h + 30)
                    (text_w, text_h), baseline = cv2.getTextSize(dominant_emotion, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, 2)
                    bg_x, bg_y = text_position[0], text_position[1] - text_h

                    # Draw rectangle
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (bg_x - 5, bg_y - 5), (bg_x + text_w + 5, bg_y + text_h + baseline + 5), (0, 0, 0), -1)
                    alpha = 0.5  # transparency factor
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # Draw text on top
                    cv2.putText(frame, dominant_emotion, text_position, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.font_colour, 2)

            cv2.imshow("Image Emotion Detection", frame)

            # Exit
            while True:
                if cv2.getWindowProperty("Image Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
                    self.stop_detection()
                    break
                if not self.detection_running:
                    break
                if cv2.waitKey(100) & 0xFF == 27:
                    break

        except Exception as e:
            print(f"DeepFace error: {e}")

        finally:
            cv2.destroyAllWindows()
            self.stop_detection()
    
    def video_emotionDetection(self):
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")]
        )
        if not video_path:
            print("Error: No file selected")
            self.stop_detection()
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            self.stop_detection()
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        frame_count = 0
        last_emotion = "Detecting..."

        try:
            while cap.isOpened() and self.detection_running:
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
                    (text_w, text_h), baseline = cv2.getTextSize(last_emotion, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, 2)
                    bg_x, bg_y = text_position[0], text_position[1] - text_h

                    # Draw background rectangle
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (bg_x - 5, bg_y - 5), (bg_x + text_w + 5, bg_y + text_h + baseline + 5), (0, 0, 0), -1)
                    alpha = 0.5
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # Draw emotion on top
                    cv2.putText(frame, last_emotion, text_position, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.font_colour, 2)

                cv2.imshow("Video Emotion Detection", frame)

                frame_count += 1

                # Exit
                if cv2.getWindowProperty("Video Emotion Detection", cv2.WND_PROP_VISIBLE) < 1:
                    self.stop_detection()
                    break
                if cv2.waitKey(30) & 0xFF == 27:  # ~30ms delay (approx 30 FPS)
                    self.stop_detection()
                    break
        finally:
            # Cleanup resources
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
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

        shown_once = False
        
        try:
            while self.detection_running:

                # Close webcam window
                if shown_once and cv2.getWindowProperty("Webcam Feed", cv2.WND_PROP_VISIBLE) < 1:
                    self.stop_detection()
                    break
                
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
                        self.update_emotion_history(emotion)
                    except Exception as e:
                        print("DeepFace error:", str(e))
                        emotion = "Error"

                    # Add background for emotion text for accessibility
                    text_position = (x, y + h + 30)
                    (text_w, text_h), baseline = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, 2)
                    bg_x, bg_y = text_position[0] ,text_position[1] - text_h
                    cv2.rectangle(frame, (bg_x - 5, bg_y - 5), (bg_x + text_w + 5, bg_y + text_h + baseline + 5), (0, 0, 0), -1)


                    # Draw emotion text
                    cv2.putText(frame, f"{emotion}", text_position, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.font_colour, 2)

                
                # Show the video feed
                cv2.imshow("Webcam Feed", frame)
                shown_once = True

                # Exit with esc
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_detection()
                    break
        finally:
            # Cleanup resources
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()


    def start_screen_emotionDetection(self):
        # Detect emotion from screen using transparent overlay
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
            while self.detection_running:

                # Close overlay window
                if cv2.getWindowProperty("EmotionLens Overlay", cv2.WND_PROP_VISIBLE) < 1:
                    self.stop_detection()
                    break
                
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
                        self.update_emotion_history(emotion)
                    except Exception as e:
                        print("DeepFace error:", str(e))
                        emotion = "Error"
                        
                    text_position = (x, y + h + 30)
                    (text_w, text_h), baseline = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, 2)
                    bg_x, bg_y = text_position[0], text_position[1] - text_h
                    
                    # Draw semi-transparent background (NOT WOKRINH)
                    overlay_rect = overlay.copy()
                    cv2.rectangle(overlay_rect, (bg_x - 5, bg_y - 5), (bg_x + text_w + 5, bg_y + text_h + baseline + 5), (0, 0, 0, 180), -1)
                    cv2.addWeighted(overlay_rect, 1.0, overlay, 1.0, 0, dst=overlay)

                    # Draw text on overlay after background
                    cv2.putText(overlay, emotion, text_position, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (*self.font_colour, 255), 2)
                
                # Display metrics on overlay
                fps = 1 / (time.time() - start_time + 1e-5)
                cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, self.text_size, (*self.font_colour, 255), 2, cv2.LINE_AA)
                
                display_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
                cv2.imshow("EmotionLens Overlay", display_overlay)

                # Exit with esc
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_detection()
                    break
        finally:
            # Cleanup resources
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
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
