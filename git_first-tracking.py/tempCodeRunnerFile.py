import cv2
import numpy as np

# Initialize the video capture (0 for webcam, or pass a video file path)
cap = cv2.VideoCapture(0)

# 1. Create the MOG2 Background Subtractor object
# detectShadows=True marks shadows in gray (value 127) instead of white (255)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=25, detectShadows=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=1)

    fg_mask = bg_subtractor.apply(frame)

    _, thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

    # Morphological opening/closing to clean up artifacts
    cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=np.ones((5, 5)))

    # 4. Find contours of the moving objects
    contours, _ = cv2.findContours(
        cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Loop through contours to draw bounding boxes around significant motion

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore tiny pixel shifts/noise
            hull = cv2.convexHull(contour, clockwise=False, returnPoints=True)
            cv2.drawContours(
                frame, [hull], contourIdx=-1, color=[255, 255, 255], thickness=2
            )
    # Display the results
    cv2.imshow("Original Feed with Tracking", frame)
    cv2.imshow("MOG2 Foreground Mask", fg_mask)
    cv2.imshow("Cleaned Binary Mask", cleaned_mask)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
