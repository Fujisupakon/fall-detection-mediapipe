import cv2
import mediapipe
import mediapipe as mp
from PIL import ImageGrab

# Set up the camera
cam = cv2.VideoCapture(0)
frame_width = 1280
frame_height = 720

# Set MediaPipe pose detector
drawing = mp.solutions.drawing_utils
person_pose = mp.solutions.pose

# Set camera resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Set display window size
cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Live Feed', 640, 380)

# Define safe zone 
safezone_x_min = 160
safezone_x_max = 480
safezone_y_min = 80
safezone_y_max = 300

num = 0
fall_detect = False
screenshot_taken = False

with person_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        
        # Resize and convert image for pose detection
        image = cv2.resize(frame, (640, 380))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        imlist = []#[[0,156,789]]

        if detect.pose_landmarks:
            drawing.draw_landmarks(frame, detect.pose_landmarks, person_pose.POSE_CONNECTIONS)

            height, width, color = frame.shape
            for i, im in enumerate(detect.pose_landmarks.landmark):
                color_x, color_y = int(im.x * width), int(im.y * height)
                imlist.append([i, color_x, color_y])

            # Draw safe zone rectangle
            cv2.rectangle(frame, (safezone_x_min, safezone_y_min), (safezone_x_max, safezone_y_max), (255, 255, 0), 2)

            if len(imlist) == 33:
                nose_x = imlist[0][1] #0,156
                nose_y = imlist[0][2]
                left_shoulder_y = imlist[11][2]
                right_shoulder_y = imlist[12][2]
                left_hip_y = imlist[23][2]
                right_hip_y = imlist[24][2]
                left_knee_y = imlist[25][2]
                right_knee_y = imlist[26][2]

                # Check if nose is outside safezone
                outside_safezone = (
                nose_x < safezone_x_min or nose_x > safezone_x_max or
                nose_y < safezone_y_min or nose_y > safezone_y_max)

                posture_fall = (
                nose_y > left_hip_y or nose_y > right_hip_y or
                nose_y > left_knee_y or nose_y > right_knee_y or
                left_shoulder_y > left_knee_y or right_shoulder_y > right_knee_y)

                # Detect fall
                if not fall_detect:
                    if posture_fall or (outside_safezone and posture_fall):
                        fall_detect = True
                
                # Reset if posture returns to normal inside safezone
                inside_safezone = (
                safezone_x_min <= nose_x <= safezone_x_max and
                safezone_y_min <= nose_y <= safezone_y_max)
                        
                # Reset condition when standing normally again
                if (nose_y < left_hip_y and nose_y < right_hip_y and
                    nose_y < left_knee_y and nose_y < right_knee_y and
                    left_shoulder_y < left_knee_y and right_shoulder_y < right_knee_y and
                    not outside_safezone):
                    fall_detect = False
                    screenshot_taken = False 

                if fall_detect:
                    cv2.putText(frame, "Fall Detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if not screenshot_taken:
                        filename = f"screenshot_{num}.png"
                        cv2.imwrite(filename, frame)
                        num += 1
                        screenshot_taken = True
                else:
                    cv2.putText(frame, "Normal", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No person detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Live Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()