from retinanet.utils.image import preprocess_image, resize_image
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
    srcs = [
        'PU_23550891_00_20200905_214516_BKQ02-003',
        'PU_23550891_00_20200905_230000_BKQ02',
    ]
    for p in srcs:
        src = os.path.join(root, p)
        os.makedirs(f'out/img/{p}', exist_ok=True)
        for f in os.listdir(src):
            frame = cv2.imread(os.path.join(src, f))
            frame, scale = resize_image(frame)
            image = preprocess_image(frame)
            boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break
                b = box.astype(int)

                frame = add_bbox(frame, b, label, ['human', 'car', 'crane'], score)
                cv2.imwrite(os.path.join(f'out/img/{p}', f), frame)
    cv2.destroyAllWindows()
