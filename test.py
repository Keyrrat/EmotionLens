import pygetwindow as gw
import pyautogui
import cv2
import numpy as np
import time

def find_teams_window():
    """
    Find the Microsoft Teams window and return its position and size.
    """
    for window in gw.getAllWindows():
        if "Microsoft Teams" in window.title:  # Adjust title as needed
            return window
    return None

# Step 1: Check if the Microsoft Teams window is open
teams_window = find_teams_window()
if not teams_window:
    print("Microsoft Teams window not found! Make sure Teams is open and visible.")
    exit()

# Step 2: Display window details
print(f"Microsoft Teams Window Found: {teams_window.title}")
print(f"Position: ({teams_window.left}, {teams_window.top})")
print(f"Size: {teams_window.width}x{teams_window.height}")

# Step 3: Capture Teams window in real time
while True:
    # Dynamically update the window position and size
    teams_window = find_teams_window()
    if not teams_window:
        print("Microsoft Teams window not found. Exiting.")
        break

    # Define the region to capture
    region = (teams_window.left, teams_window.top, teams_window.width, teams_window.height)

    # Capture the screen region
    screenshot = pyautogui.screenshot(region=region)

    # Convert the screenshot to an OpenCV-compatible format
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the captured frame
    cv2.imshow("Microsoft Teams Capture", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Optional: Limit frame rate to reduce CPU usage
    time.sleep(0.1)  # Capture ~10 frames per second

# Cleanup
cv2.destroyAllWindows()