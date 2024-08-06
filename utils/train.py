import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def accuracy_score(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    correct = (y_pred == y_true).float().sum()
    total = y_true.numel()
    accuracy = correct / total
    return accuracy.item()

def plot_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list):
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    ax[0].plot(train_loss_list, color='blue', label=f'Training Loss (Min: {min(train_loss_list):.3f})')
    ax[0].plot(val_loss_list, color='red', label=f'Validation Loss (Min: {min(val_loss_list):.3f})')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Training vs Validation Loss')

    ax[1].plot(train_acc_list, color='blue', label=f'Training Accuracy (Max: {max(train_acc_list):.3f})')
    ax[1].plot(val_acc_list, color='red', label=f'Validation Accuracy (Max: {max(val_acc_list):.3f})')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].set_title('Training vs Validation Accuracy')

    plt.tight_layout()
    plt.show()

def training(model, loss_fn, optimizer, train_loader, val_loader, epochs, device):
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_val_accuracy = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in tqdm(train_loader):
            Xs, ys = batch
            Xs = Xs.to(device)
            ys = ys.to(device)

            model.zero_grad()
            preds = model(Xs)
            loss = loss_fn(preds, ys)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN values detected in parameter {name}")

            # Calculate accuracy
            with torch.no_grad():
                accuracy = accuracy_score(ys, preds)
                epoch_accuracy += accuracy

        average_loss = epoch_loss / len(train_loader)
        average_accuracy = epoch_accuracy / len(train_loader)

        print(f"Epoch {epoch+1} average loss: {average_loss:.3f}, average accuracy: {average_accuracy:.3f}", end=" ")

        train_loss_list.append(average_loss)
        train_acc_list.append(average_accuracy)

        # loss and accuracy on the validation set
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                Xs, ys = batch
                Xs = Xs.to(device)
                ys = ys.to(device)

                preds = model(Xs)
                loss = loss_fn(preds, ys)
                val_loss += loss.item()

                # Calculate accuracy
                val_accuracy += accuracy_score(ys, preds)

        val_average_loss = val_loss / len(val_loader)
        val_average_accuracy = val_accuracy / len(val_loader)

        print(f"average val loss: {val_average_loss:.3f}, average val accuracy: {val_average_accuracy:.3f}")

        val_loss_list.append(val_average_loss)
        val_acc_list.append(val_average_accuracy)
        
        if best_val_accuracy < val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"Saving model with new best validation accuracy: {best_val_accuracy:.4f}")
            torch.save(model.state_dict(), 'model_weights/Resnet50_BCEDiceLoss_256_128ov_100epoch.pth')

    plot_loss_acc(train_loss_list, train_acc_list, val_loss_list, val_acc_list)

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list
