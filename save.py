import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from retinanet.preprocessing.csv_generator import CSVGenerator
from tensorflow import keras
from retinanet import losses
from retinanet import models
from retinanet.models.retinanet import retinanet_bbox
from retinanet.utils.config import parse_anchor_parameters
from retinanet.utils.model import freeze as freeze_model
import cv2
import numpy as np
import tensorflow as tf

color_list = np.array([0.466, 0.750, 0.325, 0.850, 0.098, 1.000, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556,
                       1.000, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078, 0.184, 0.300, 0.300, 0.300,
                       0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
                       0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.667, 0.000,
                       0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000,
                       1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500,
                       0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500,
                       0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500,
                       0.667, 0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500,
                       1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000,
                       0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000, 0.333, 0.667, 1.000,
                       0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000,
                       0.667, 1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000,
                       0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
                       0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
                       0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000,
                       0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667,
                       0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143,
                       0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714,
                       0.857, 0.857, 0.857, 0.000, 0.447, 0.741, 0.50, 0.5, 0]).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
color_list = np.concatenate((color_list, color_list/2))

colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)


def add_bbox(img, bbox, cat, labels, conf=1, show_txt=True, color=None):
    # bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    c = colors[cat % 158][0][0].tolist() if color is None else color
    txt = '{} - {:.1f}'.format(labels[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
        cv2.rectangle(img,
                      (bbox[0], bbox[1] - cat_size[1] - 2),
                      (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
        cv2.putText(img, txt, (bbox[0], bbox[1] - 2),
                    font, 0.5, (0, 0, 0) if cat < 80 else (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    return img


def create_generator(annotations, classes, image_min_side, image_max_side):
    """ Create generators for evaluation.
    """
    validation_generator = CSVGenerator(
        annotations,
        classes,
        image_min_side=image_min_side,
        image_max_side=image_max_side,
    )
    return validation_generator


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

    # compile model
    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model



if __name__ == '__main__':
    # model = load_model('/home/palm/PycharmProjects/mine/snapshots/infer_model_test.h5', backbone=backbone, submodels=submodels)
    backbone = models.backbone('resnet50')

    labels_to_names = {0: 'car', 1: 'crane', 2: 'human'}
    main_model, training_model, prediction_model = create_models(backbone.retinanet, 3)
    prediction_model.load_weights('/home/palm/PycharmProjects/mine/snapshots/infer_model_test.h5')
    # image = Image.open('/media/palm/BiggerData/mine/0.jpg')
    # image = cv2.resize(np.array(image), (800, 450))
    # image = preprocess_image(image)
    # image = np.expand_dims(image, axis=0)
    # data = prediction_model.predict_on_batch(image)
    # print(data)

    tf.keras.models.save_model(
        prediction_model,
        'weights/1',
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
