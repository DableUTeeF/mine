# Weight file
https://drive.google.com/drive/folders/15n_vorI-v_U14fgPwzzKhNlOD14_bT_g?usp=sharing

# In case compiled C++ doesn't work

https://github.com/facebookresearch/detr

https://github.com/fizyr/keras-retinanet

# Install the server
```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list 
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt update
sudo apt install tensorflow-model-server
```
# Run
```
nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=fashion_model \
  --model_base_path='/home/palm/PycharmProjects/test_serving/weights' >server.log 2>&1

```
