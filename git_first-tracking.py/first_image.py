import cv2

cap = cv2.VideoCapture(0)
while True:
    return_val, frame = cap.read()
    if return_val == False:
        print("camera cant be accessed")
        break
    frame_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_3 = cv2.resize(frame_2, (800, 600), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("sigma", frame_3)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
