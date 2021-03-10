from flask import Flask
from flask import request, render_template

import numpy as np
import os

import boto3
import torch
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
import albumentations
from download_models import download

from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader
from torch.utils.data import DataLoader

from src.utils.resize_images import resize_image
from src.config.config import MEAN, STD, DEVICE, FOLDS

app = Flask(__name__)
UPLOAD_PATH = 'static/image'
MODEL = None
MODEL_PATH = "model_s3"

if not os.path.exists(MODEL_PATH):
    download()

if not torch.cuda.is_available():
    DEVICE = "cpu"

def test_dataloader(images, targets):
    dataset = ClassificationLoader(
        image_paths=images,
        targets=targets,
        resize=None,
        augmentations=albumentations.Compose(
        [
            albumentations.Normalize(MEAN, STD, max_pixel_value=255.0, always_apply=True)
        ])
    )
    data_loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=0
    )
    return data_loader


class ResNext101_64x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(ResNext101_64x4d, self).__init__()
        self.model = pretrainedmodels.resnext101_64x4d(pretrained=pretrained)
        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(1, -1)
        out = torch.sigmoid(self.l0(x))
        loss = 0
        return out, loss


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


def predict_result(image_path, model):

    test_images = [image_path]
    target = [0]

    test_loader = test_dataloader(images=test_images, targets=target)

    predictions = Engine.predict(test_loader, model=model, device=DEVICE)
    return np.vstack((predictions)).ravel()


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        data = request.files['image']
        if data:
            image_location = os.path.join(
                UPLOAD_PATH,
                "img.jpg"
            )
            data.save(image_location)
            
            resize_image(filepath=image_location, output_folder=UPLOAD_PATH, resize=[224, 224])

            model = []
            for i in range(5):
                MODEL = ResNext101_64x4d(pretrained=None)
                MODEL.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"model_fold_{i}.bin")))
                MODEL.to(device=DEVICE)
                model.append(MODEL)
            
            p1 = predict_result(image_location, model[0])
            p2 = predict_result(image_location, model[1])
            p3 = predict_result(image_location, model[2])
            p4 = predict_result(image_location, model[3])
            p5 = predict_result(image_location, model[4])
            
            pred = p1+p2+p3+p4+p5 / 5

            return render_template("index.html", predicted_label= "Melanoma Detected" if pred >= 0.5 else "Melanoma not detected")
    return render_template("index.html", predicted_label="select an image")


if __name__ == "__main__":
    app.run(debug=True, port=12000)