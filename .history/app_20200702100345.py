import os
from flask import Flask, request, redirect, url_for, jsonify, Response

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

from PIL import Image
import requests
from io import BytesIO


app = Flask(__name__)
model = None

def load_model():
    global model
    model = VGG16(weights='imagenet', include_top=True)

@app.route('/predict', methods=['GET'])
def upload_file():
    response = {'success': False}
    img_url = request.args.getlist('url')
    response['predictions'] = []
    for url in img_url:
        responsePredict = {'url': url}
        result = requests.get(url)
        img = Image.open(BytesIO(result.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        inputs = preprocess_input(img)

        preds = model.predict(inputs)
        results = decode_predictions(preds)

        responsePredict['predictions'] = []
        for (imagenetID, label, prob) in results[0]: # [0] as input is only one image
            row = {'label': label, 'probability': float(prob)} # numpy float is not good for json
            responsePredict['predictions'].append(row)
        response['success'] = True
    return jsonify(response)

if __name__ == '__main__':
    load_model()
    app.run(threaded=False)