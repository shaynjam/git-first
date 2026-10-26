import cv2

cap = cv2.VideoCapture(0)  # 1. Turn on your default webcam

while True:
    return_val, frame = cap.read()  # 2. Grab the current frame from the camera
    if not return_val:
        break  # If the camera fails to grab a frame, exit the loop

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )  # 3. Convert the frame to Black & White
    mirror = cv2.flip(gray, 1)  # 4. Flip it horizontally so it behaves like a mirror
    mirror_2 = cv2.resize(mirror, (500, 500), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("My B&W Mirror Project", mirror_2)  # 5. Show the result in a window

    if cv2.waitKey(1) & 0xFF == ord("q"):  # 6. Stop if you press the 'q' key
        break

cap.release()  # 7. Turn off the webcam hardware
cv2.destroyAllWindows()  # 8. Close the popup window
