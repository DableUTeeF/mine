from retinanet.utils.image import preprocess_image, resize_image, read_image_bgr
import numpy as np
import cv2
import tensorflow as tf
import keras
from retinanet.boxutils import add_bbox
import os
from retinanet import losses
from retinanet import models
from retinanet.utils.model import freeze as freeze_model
from retinanet.models.retinanet import retinanet_bbox
from retinanet.utils.config import parse_anchor_parameters

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

def create_models(backbone_retinanet, num_classes, freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model

    model = backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier)
    training_model = model
    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    return model, training_model, prediction_model


if __name__ == '__main__':
    backbone = models.backbone('resnet50')

    labels_to_names = {0: 'car', 1: 'crane', 2: 'human'}
    main_model, training_model, prediction_model = create_models(backbone.retinanet, 3)
    prediction_model.load_weights('snapshots/infer_model_test.h5')

    # prediction_model = models.load_model('snapshots/infer_model_test.h5')
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
