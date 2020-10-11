from retinanet.utils.image import preprocess_image
from retinanet import models
import numpy as np
import cv2
import tensorflow as tf
import keras
import os
from retinanet.boxutils import add_bbox
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)


if __name__ == '__main__':
    prediction_model = models.load_model('snapshots/infer_model_test.h5')
    # prediction_model.summary()
    # exit()
    """
    
    """

    for file in os.listdir('/media/palm/BiggerData/mine/new/v/'):
        cap = cv2.VideoCapture(os.path.join('/media/palm/BiggerData/mine/new/v/', file))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(f'/media/palm/BiggerData/mine/out/{file}.mp4', fourcc, 20.0, (1333, 750))
        f = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (1333, 750))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            image = preprocess_image(frame)
            boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break
                b = box.astype(int)

                frame = add_bbox(frame, b, label, ['crane', 'car', 'human'], score)
            out.write(frame)

        cap.release()
        out.release()
    cv2.destroyAllWindows()
