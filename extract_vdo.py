import cv2
import os
dst = '/media/palm/BiggerData/mine'

cap = cv2.VideoCapture('/home/palm/Videos/PU_22788397_00_20200423_110031_BKQ02.mkv')
f = 0
while cap.isOpened():
    ret, frame = cap.read()
    if f % 60 == 0:
        cv2.imwrite(os.path.join(dst, str(f)+'.jpg'), frame)
    f += 1


cap.release()
cv2.destroyAllWindows()
