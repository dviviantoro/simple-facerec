import cv2
import time
from datetime import datetime
import os
import json

video_path = '/Users/deny/optimized-facerec/assets/marathon.mp4'
archive_path = '/Users/deny/optimized-facerec/archive/raw/'
timedef = "/Users/deny/optimized-facerec/assets/timedef.json"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # capture 1 frame per second
frame_count = 0
count = 0
last_target_dir = ""

def get_timestamp():
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%dT%H:%M:%S")
    return current_time

def is_time_in_range(start_time, end_time):
    current_time = datetime.now().time()
    if start_time <= current_time <= end_time:
        return True
    else:
        return False

def dir_path():
    with open(timedef, 'r') as file:
        data = json.load(file)

    for i in data:
        start = datetime.strptime(i['start'], "%H:%M:%S").time()
        end = datetime.strptime(i['end'], "%H:%M:%S").time()
        
        if is_time_in_range(start, end):
            my_id = i["id"]
            now = datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            target_dir = f"/Users/deny/optimized-facerec/archive/{current_date}id{my_id}"
            try:
                os.makedirs(target_dir)
            except Exception as e:
                # print(e)
                pass
            finally:
                return target_dir

# ocv font setting
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255, 255, 255)
thickness = 1
position = (10, 30)

while True:
    timestamp = get_timestamp()
    target_dir = dir_path()
    ret, frame = cap.read()
    
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_interval == 0:
        if target_dir != last_target_dir : count = 0
        print(target_dir)
        print(last_target_dir)

        count += 1
        formatted_count = "{:03d}".format(count)

        cv2.putText(frame, timestamp, position, font, font_scale, color, thickness)
        cv2.imshow('Captured Frame', frame)   
        cv2.imwrite(f"{target_dir}/frame{formatted_count}.jpg", frame)
        print(f"Captured frame {frame_count}")

        last_target_dir = target_dir

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()