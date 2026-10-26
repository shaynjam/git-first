import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, prev = cap.read()

last_frame = cv2.resize(prev, (1024, 768), interpolation=cv2.INTER_AREA)
last_frame = cv2.flip(last_frame, 1)

secondlast_frame = None
count = 0

while True:
    return_val, frame = cap.read()

    if return_val is not True:
        print("cant access camera")
        break

    frame2 = cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_AREA)
    frame2 = cv2.flip(frame2, 1)
    frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), sigmaX=1)

    if count < 2:
        display_frame = frame2

    else:

        Current_difference = cv2.absdiff(frame_gray, last_frame)
        previous_difference = cv2.absdiff(last_frame, secondlast_frame)

        _, Current_difference = cv2.threshold(
            Current_difference, 20, maxval=255, type=cv2.THRESH_BINARY
        )
        _, previous_difference = cv2.threshold(
            previous_difference, 20, maxval=255, type=cv2.THRESH_BINARY
        )

        pre_dilation = cv2.bitwise_and(Current_difference, previous_difference)

        dilated_frame = cv2.dilate(pre_dilation, kernel=np.ones((5, 5)), iterations=1)

        morphed_frame = cv2.morphologyEx(
            dilated_frame, cv2.MORPH_CLOSE, kernel=np.ones((5, 5)), iterations=3
        )

        cv2.imshow("post dilation and morph", morphed_frame)

        contours, heirarchys = cv2.findContours(
            morphed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        frame2_copy = frame2.copy()

        if len(contours) > 0:
            for contour in contours:
                if cv2.contourArea(contour) > 30:
                    hull = cv2.convexHull(contour, clockwise=False, returnPoints=True)

                    cv2.drawContours(
                        frame2_copy,
                        [hull],
                        contourIdx=-1,
                        color=(255, 255, 255),
                        thickness=2,
                    )

        display_frame = frame2_copy

    secondlast_frame = last_frame
    last_frame = frame_gray

    count = count + 1
    cv2.imshow("Motion Tracker", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print(count)

cap.release()
cv2.destroyAllWindows()
