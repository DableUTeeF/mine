import torch
from detr.models.detr import DETR, PostProcess
from detr.models.position_encoding import PositionEmbeddingLearned, PositionEmbeddingSine
from detr.models.backbone import Backbone, Joiner, MNetBackbone
from detr.models.transformer import Transformer
from PIL import Image
import torchvision.transforms as T
from detr.evaluate_util import add_bbox, all_annotation_from_instance, create_csv_training_instances
import time
import cv2
import numpy as np
import os


def make_coco_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.Resize(800),
        normalize,
    ])


def build_position_encoding(hidden_dim, position_embedding):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")
    return position_embedding


def build_backbone(lr_backbone, masks, backbone, dilation, hidden_dim, position_embedding):
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    train_backbone = lr_backbone > 0
    return_interm_layers = masks
    if 'resnet' in backbone:
        backbone = Backbone(backbone, train_backbone, return_interm_layers, dilation)
    elif 'mobilenet' in backbone:
        backbone = MNetBackbone(train_backbone, return_interm_layers)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes, fname):
    pil_img = np.array(pil_img)
    pil_img = pil_img[..., ::-1]
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cat = p.argmax()
        pil_img = add_bbox(pil_img, (xmin, ymin, xmax, ymax), cat, ['Nope', 'car', 'crane', 'human'], p[cat])
    # cv2.imshow('t', pil_img)
    # cv2.waitKey()
    # cv2.imwrite(fname, pil_img)
    return pil_img


if __name__ == '__main__':
    hidden_dim = 256
    position_embedding = 'sine'
    lr_backbone = 1e-5
    masks = False
    backbone = 'resnet50'
    dilation = False
    dropout = 0.1
    nheads = 8
    dim_feedforward = 2048
    enc_layers = 6
    dec_layers = 6
    pre_norm = False
    num_classes = 4
    num_queries = 100
    aux_loss = True

    backbone = build_backbone(lr_backbone, masks, backbone, dilation, hidden_dim, position_embedding)
    transformer = build_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm)
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=aux_loss,
    )
    transform = make_coco_transforms()
    postprocessors = PostProcess()

    checkpoint = torch.load('snapshots/mine/checkpoint.pth')
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(
        '/home/palm/PycharmProjects/mine/anns/ann.csv',
        '/home/palm/PycharmProjects/mine/anns/ann.csv',
        '/home/palm/PycharmProjects/mine/anns/c.csv',
    )
    names = [o['filename'] for o in train_ints]
    # os.listdir()
    # all_detections = []
    # all_annotations = []
    # # todo: algea
    # for instance in valid_ints:
    #     t = time.time()
    #     all_annotation = all_annotation_from_instance(instance)
    #     target_image_ori = Image.open(instance["filename"])
    #     target_image = transform(target_image_ori)
    #     x = torch.zeros((1, *target_image.shape))
    #     # x[0] = target_image
    #     outputs = model(x.cuda())
    #     probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    #     keep = probas.max(-1).values > 0.7
    #
    #     # convert boxes from [0; 1] to image scales
    #     bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), target_image_ori.size).int().cpu().detach().numpy()
    #     # p = probas[keep]
    #     print(time.time() - t)
    #     # plot_results(target_image_ori, p, bboxes_scaled, os.path.join('predict', os.path.basename(instance["filename"])))

    # todo: mine
    root = '/media/palm/BiggerData/mine/new/i/'
    srcs = os.listdir(root)
    for pp in srcs:
        src = os.path.join(root, pp)
        os.makedirs(f'predict/mine/{pp}', exist_ok=True)
        for f in os.listdir(src):
            print(os.path.join(root, pp, f))
            t = time.time()
            target_image_ori = Image.open(os.path.join(root, pp, f)).convert('L').convert('RGB')
            target_image = transform(target_image_ori)
            x = torch.zeros((1, *target_image.shape))
            x[0] = target_image
            outputs = model(x.cuda())
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.7

            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), target_image_ori.size).int().cpu().detach().numpy()
            p = probas[keep]
            print(time.time() - t)
            cv2_image = plot_results(target_image_ori, p, bboxes_scaled, os.path.join(f'predict/mine/{pp}', f))
            if os.path.join(root, pp, f) not in names:
                cv2.imwrite(os.path.join(f'predict/mine/{pp}', f), cv2_image)

    # # todo: one pic
    # target_image_ori= Image.open('/media/palm/BiggerData/mine/new/i/PU_23550891_00_20200905_230000_BKQ02/120.jpg').convert('L').convert('RGB')
    # target_image = transform(target_image_ori)
    # x = torch.zeros((1, *target_image.shape))
    # x[0] = target_image
    # outputs = model(x.cuda())
    # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # keep = probas.max(-1).values > 0.5
    #
    # # convert boxes from [0; 1] to image scales
    # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), target_image_ori.size).int().cpu().detach().numpy()
    # p = probas[keep]
    # plot_results(target_image_ori, p, bboxes_scaled, os.path.join('test.jpg'))
