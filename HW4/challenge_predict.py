'''
Challenge - Predict
    Runs the challenge model inference on the test dataset and saves the
    predictions to disk
    Usage: python predict_challenge.py
'''
import argparse
import numpy as np
import pandas as pd
from data import get_train_val_test_loaders
from challenge_model import Challenge
from utils import *
import utils

def predict_challenge(data_loader, model):
    """
    Runs the model inference on the test set and outputs the predictions
    """
    model_pred = np.array([])
    for i, (X, y) in enumerate(data_loader):
        output = model(X)
        predicted = predictions(output.data)
        predicted = predicted.cpu().numpy()
        model_pred = np.concatenate([model_pred, predicted])
    return model_pred

def main():
    device = utils.get_device()

    # data loaders
    _, _, te_loader, get_semantic_label = get_train_val_test_loaders(num_classes=config('challenge.num_classes'))

    # Attempts to restore the latest checkpoint if exists
    model = Challenge(0)
    model, _, _ = restore_checkpoint(model, config('challenge.checkpoint'))
    model = model.to(device)
    model.eval()
    
    # Evaluate model
    model_pred = predict_challenge(te_loader, model)

    print('saving challenge predictions...\n')
    model_pred = [get_semantic_label(p) for p in model_pred]
    pd_writer = pd.DataFrame(model_pred, columns=['predictions'])
    pd_writer.to_csv('predictions.csv', index=False, header=False)

if __name__ == '__main__':
    main()
