import cv2

# Open the video file
video_path = '/Users/deny/optimized-facerec/assets/marathon.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video.")
    exit()

# Loop to read and display frames
while True:
    ret, frame = cap.read()

    # If frame was read successfully
    if ret:
        # Display the frame
        cv2.imshow('Video Playback', frame)

        # Wait for 1 ms and check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
