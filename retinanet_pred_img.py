from retinanet.utils.image import preprocess_image, resize_image, read_image_bgr
from retinanet import models
import numpy as np
import cv2
import tensorflow as tf
import keras
from retinanet.boxutils import add_bbox
import os

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

if __name__ == '__main__':
    prediction_model = models.load_model('snapshots/infer_model_test.h5')
    root = '/media/palm/BiggerData/mine/new/i/'
    # srcs = [
    #     'PU_23550891_00_20200905_214516_BKQ02-003',
    #     'PU_23550891_00_20200905_230000_BKQ02',
    # ]
    srcs = os.listdir(root)
    for p in srcs:
        src = os.path.join(root, p)
        os.makedirs(f'/media/palm/BiggerData/mine/out/i/{p}', exist_ok=True)
        for f in os.listdir(src):
            frame = read_image_bgr(os.path.join(src, f))

            im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.dstack((im, im, im))
            frame, scale = resize_image(frame, min_side=720, max_side=1280)
            image = preprocess_image(frame)
            boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break
                b = box.astype(int)

                frame = add_bbox(frame, b, label, ['crane', 'car', 'human', ], score)
                cv2.imwrite(os.path.join(f'/media/palm/BiggerData/mine/out/i/{p}', f), frame)
    cv2.destroyAllWindows()
