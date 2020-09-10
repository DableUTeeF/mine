import cv2
import os
dst = '/media/palm/BiggerData/mine/new/i'
src = '/media/palm/BiggerData/mine/new/v'
for file in os.listdir(src):
    os.makedirs(os.path.join(dst, file[:-4]))
    cap = cv2.VideoCapture(os.path.join(src, file))
    f = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if f % 60 == 0:
                cv2.imwrite(os.path.join(dst, file[:-4], str(f)+'.jpg'), frame)
            f += 1
    except Exception as e:
        print(e)
    finally:
        cap.release()
cv2.destroyAllWindows()
