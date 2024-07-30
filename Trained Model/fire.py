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
    # Gave path to your sound file
    playsound("")

# Function to send email alert
def sendEmailFunc(recipientEmails):
    subject =  "Fire Alert at ABC University"
    message = f"Subject: {subject}\n\n  Emergency Fire Alert: ABC University\n\n A fire has been detected. Follow these steps immediately:\n - Evacuate: Leave the building using the nearest exit. Do not use elevators.\n - Alert Others: Inform people around you.\n - Avoid Smoke: Stay low and cover your nose and mouth if needed.\n - Assembly Points: Go to Main Ground, Parking Lot , or the Sports Field.\n - Assist: Help those needing assistance.\n - Stay Informed: Await further instructions from officials.\n - Do Not Re-enter: Stay out until cleared by authorities.\n\n  University of Chakwal Emergency Management Team"

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        # Add Your Gmail account also 2 step verification should be enabled 
        server.login("your gmail", "pass")
        for recipientEmail in recipientEmails:
            server.sendmail('your gmail', recipientEmail, message)
            print("Email sent to {}".format(recipientEmail))
        server.quit()
    except Exception as e:
        print(e)

# Initialize the YOLO model
model = YOLO('best.pt')

# Reading the classes
classnames = ['fire']

# Open the image file (change path as needed)
frame = cv2.imread("")#Add any fire image to test 
if frame is None:
    print("Error: Could not read image.")
    exit()

# Process the image
frame = cv2.resize(frame, (640, 480))
result = model(frame, stream=True)

# Process the YOLO model results
for info in result:
    boxes = info.boxes
    for box in boxes:
        confidence = box.conf[0]
        confidence = math.ceil(confidence * 100)
        Class = int(box.cls[0])
        if confidence > 50:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                               scale=1.5, thickness=2)

            # Fire detection logic
            Fire_Detected += 1
            if Fire_Detected >= 1 and not alarm_status:
                alarm_status = True
                # List of recipient email addresses
                # Add your recipient email that you have to send mail
                recipientEmails = ["email1","mail2","etc"]
                # Send email alert and play sound in separate threads
                threading.Thread(target=sendEmailFunc, args=(recipientEmails,)).start()
                threading.Thread(target=play_audio).start()

# Display the annotated image
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()