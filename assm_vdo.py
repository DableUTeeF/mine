from retinanet.utils.image import preprocess_image
from retinanet import models
import numpy as np
import cv2
import tensorflow as tf
import keras
from retinanet.boxutils import add_bbox
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)


if __name__ == '__main__':
    prediction_model = models.load_model('snapshots/infer_model_test.h5')
    cap = cv2.VideoCapture('/media/palm/BiggerData/mine/new/v/PU_23550891_00_20200905_203537_BKQ02-005.mkv')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out/output.avi', fourcc, 20.0, (800, 450))
    f = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (800, 450))
        image = preprocess_image(frame)
        boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
            b = box.astype(int)

            frame = add_bbox(frame, b, label, ['human', 'car', 'pole'], score)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
