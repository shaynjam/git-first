import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, prev = cap.read()

last_frame = cv2.resize(prev, (1024, 768), interpolation=cv2.INTER_AREA)
last_frame = cv2.flip(last_frame, 1)

secondlast_frame = None
count = 0
kernel = np.ones((3, 3), np.uint8)

while True:
    return_val, frame = cap.read()
    if return_val is not True:
        print("cant access camera")
        break

    frame2 = cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_AREA)
    frame2 = cv2.flip(frame2, 1)
    frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (0, 0), sigmaX=1)

    if count < 2:
        display_frame = frame2

    else:

        difference_1 = cv2.absdiff(frame_gray, last_frame)
        difference_2 = cv2.absdiff(last_frame, secondlast_frame)

        retu, difference_1 = cv2.threshold(
            difference_1, 35, maxval=255, type=cv2.THRESH_BINARY
        )
        retu_2, difference_2 = cv2.threshold(
            difference_2, 35, maxval=255, type=cv2.THRESH_BINARY
        )

        pre_dilation = cv2.bitwise_and(difference_1, difference_2)
        pre_dilation = cv2.dilate(
            pre_dilation, kernel=np.ones((9, 9), np.uint8), iterations=30
        )
        pre_dilation = cv2.morphologyEx(
            pre_dilation, cv2.MORPH_GRADIENT, kernel=np.ones((9, 9))
        )

        contours, heirarchy = cv2.findContours(
            pre_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_frame = None
        frame2_copy = frame2.copy()
        for contour in contours:
            if cv2.contourArea(contour) >= 3000:
                x, y, w, h = cv2.boundingRect(contour)
                contour_frame = cv2.rectangle(
                    frame2_copy, (x, y), (x + w, y + h), (0, 0, 255)
                )

        if contour_frame is not None:
            display_frame = contour_frame
        else:
            display_frame = frame2
    secondlast_frame = last_frame
    last_frame = frame_gray

    count = count + 1
    cv2.imshow("Motion Tracker", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print(count)

cap.release()
cv2.destroyAllWindows()
