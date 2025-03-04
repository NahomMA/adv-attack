from cfg import cfg
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np


def test(model, test_loader, device, model_load_path=None):
    if model_load_path:
        checkpoint = torch.load(model_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_acc = correct / total
    
    # Convert lists to numpy arrays 
    return test_acc, np.array(all_predictions),np.array(all_targets)