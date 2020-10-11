import requests
import json
from PIL import Image
import numpy as np
import cv2
from retinanet.utils.image import preprocess_image

headers = {"content-type": "application/json"}

if __name__ == '__main__':
    image = Image.open('/media/palm/BiggerData/mine/new/i/PU_23550891_00_20200905_203537_BKQ02-005/0.jpg')
    image = cv2.resize(np.array(image), (800, 450))
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)
    data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})
    json_response = requests.post('http://localhost:8501/v1/models/retinanet:predict', data=data, headers=headers)
    x = json.loads(json_response.text)
    # print(x)
    boxes = x['predictions'][0]['filtered_detections']
    scores = x['predictions'][0]['filtered_detections_1']
    labels = x['predictions'][0]['filtered_detections_2']
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        b = box.astype(int)
        print()
