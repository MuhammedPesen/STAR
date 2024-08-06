import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

def dice_coefficient(outputs, targets, smooth=1e-6):
    preds = (outputs > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    train_losses = []
    dice_scores = []
    
    for batch in tqdm(train_loader):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        with torch.no_grad():
            dice_score = dice_coefficient(outputs, targets)
            dice_scores.append(dice_score)

    avg_loss = np.mean(train_losses)
    avg_dice_score = np.mean(dice_scores)
    
    return avg_loss, avg_dice_score

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_losses = []
    dice_scores = []
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            val_losses.append(loss.item())
            
            dice_score = dice_coefficient(outputs, targets)
            dice_scores.append(dice_score)

    avg_loss = np.mean(val_losses)
    avg_dice_score = np.mean(dice_scores)
    
    return avg_loss, avg_dice_score

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path, patience=10):
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_dice_coefficient': [],
        'val_loss': [],
        'val_dice_coefficient': []
    }
    
    best_dice_coefficient = 0
    epochs_no_improve = 0
    min_val_loss = np.inf
    
    for epoch in range(epochs):
        train_loss, train_dice_coefficient = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice_coefficient = validate_epoch(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Dice Score: {train_dice_coefficient:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Dice Score: {val_dice_coefficient:.4f}')
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_dice_coefficient'].append(train_dice_coefficient)
        metrics['val_loss'].append(val_loss)
        metrics['val_dice_coefficient'].append(val_dice_coefficient)
                
        if val_dice_coefficient > best_dice_coefficient:
            epochs_no_improve = 0
            best_dice_coefficient = val_dice_coefficient
            print(f"Saving model with new best validation dice score: {best_dice_coefficient:.4f}")
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    return metrics
