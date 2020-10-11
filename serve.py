import requests
import json
from PIL import Image
import numpy as np
import cv2
from retinanet.utils.image import preprocess_image


image = Image.open('/media/palm/BiggerData/mine/new/i/PU_23550891_00_20200905_203537_BKQ02-005/0.jpg')
image = cv2.resize(np.array(image), (800, 450))
image = preprocess_image(image)
image = np.expand_dims(image, axis=0)
data = json.dumps({"signature_name": "serving_default", "instances": image.tolist()})

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/fashion_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)
print(predictions)
