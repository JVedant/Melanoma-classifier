import os
import numpy as np
import pandas as pd

import torch
from sklearn import metrics

from model import ResNext101_64x4d
from config.config import *

from utils.image_loader import train_dataloader, valid_dataloader

import pretrainedmodels

from wtfml.utils import EarlyStopping
from wtfml.engine import Engine

if not torch.cuda.is_available:
    DEVICE = "cpu"

def train(fold):
    training_data_path = TRAINING_DATA_PATH
    df = pd.read_csv(TRAIN_FOLDS)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = ResNext101_64x4d(pretrained="imagenet")
    model.to(DEVICE)

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_loader = train_dataloader(images=train_images, targets=train_targets)
    valid_loader = valid_dataloader(images=valid_images, targets=valid_targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(patience=3, mode="max")

    for epoch in range(EPOCHS):
        train_loss = Engine.train(train_loader, model, optimizer, device=DEVICE)
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=DEVICE
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}")
        scheduler.step(auc)

        es(auc, model, model_path=os.path.join(MODEL_PATH, f"model_fold_{fold}.bin"))
        if es.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":

    for fold in range(FOLDS):
        train(fold)