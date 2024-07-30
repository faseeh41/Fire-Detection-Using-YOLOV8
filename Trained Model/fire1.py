from ultralytics import YOLO
import cv2
import cvzone
import math
from playsound import playsound
import smtplib
import threading

# Alarm and email status variables
Fire_Detected = 0
alarm_status = False

# Function to play the alarm sound
def play_audio():
    try:
        # Gave path to your sound file
        playsound("")
    except Exception as e:
        print(f"Error playing sound: {e}")

# Function to send email alert
def sendEmailFunc(recipientEmails):
    subject = "Fire Alert at ABC University"
    message = f"Subject: {subject}\n\nEmergency Fire Alert: University of ABC\n\nA fire has been detected. Follow these steps immediately:\n- Evacuate: Leave the building using the nearest exit. Do not use elevators.\n- Alert Others: Inform people around you.\n- Avoid Smoke: Stay low and cover your nose and mouth if needed.\n- Assembly Points: Go to Main Ground, Parking Lot, or the Sports Field.\n- Assist: Help those needing assistance.\n- Stay Informed: Await further instructions from officials.\n- Do Not Re-enter: Stay out until cleared by authorities.\n\nUniversity of Chakwal Emergency Management Team"

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        # Login TO Your Gmail Add your AGMAIL ACCount also 2 step verfication should be enabled to it
        server.login("Your Gmail", "Pss")
        for recipientEmail in recipientEmails:
            server.sendmail('Your gmail', recipientEmail, message)
            print("Email sent to {}".format(recipientEmail))
        server.quit()
    except Exception as e:
        print(f"Error sending email: {e}")

# Initialize the YOLO model
model = YOLO('best.pt')

# Reading the classes
classnames = ['fire']

# Open the video file or webcam
video = cv2.VideoCapture(0)  # 0 for webcam, or specify the path to the video file

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

def process_frame(frame):
    global Fire_Detected, alarm_status

    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Process the YOLO model results
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 20:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)

                # Fire detection logic
                Fire_Detected += 1
                if Fire_Detected >= 1 and not alarm_status:
                    alarm_status = True
                    # List of recipient email addresses
                    recipientEmails = ["email1","email2","etc"]
                    # Send email alert and play sound in separate threads
                    threading.Thread(target=sendEmailFunc, args=(recipientEmails,)).start()
                    threading.Thread(target=play_audio).start()

    return frame

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    frame = process_frame(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video.release()
cv2.destroyAllWindows()
