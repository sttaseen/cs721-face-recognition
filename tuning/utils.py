import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# Set device type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_class_weights(dataset):
    """
    Compute class weights for imbalanced classes.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset from which to compute class weights.
    
    Returns:
        torch.Tensor: A tensor of class weights.
    """
    class_indices = dataset.targets if hasattr(dataset, 'targets') else np.array([y for _, y in dataset])
    class_counts = torch.bincount(torch.tensor(class_indices))
    total_count = len(class_indices)
    class_weights = total_count / (len(np.unique(class_indices)) * class_counts.float())
    return class_weights

def create_criterion(class_weights):
    """Returns a criterion with class weights."""
    class_weights = class_weights.to(device)  # Ensure class weights are on the same device as the model
    return nn.CrossEntropyLoss(weight=class_weights)


def train_model(model, train_dataloader, val_dataloader, num_epochs=50, lr=0.001, patience=5):
    """
    Train a model with early stopping based on validation loss.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        patience (int): Number of epochs with no improvement to wait before stopping.

    Returns:
        nn.Module: The trained model with the best weights.
    """
    model.to(device)
    
    # Compute class weights and create criterion
    class_weights = compute_class_weights(train_dataloader.dataset)
    criterion = create_criterion(class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_dataloader.dataset)
        val_loss = evaluate_loss(model, val_dataloader, criterion)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    print('Training complete')
    model.load_state_dict(best_model_wts)
    return model

def evaluate_loss(model, dataloader, criterion):
    """
    Evaluate the model's loss on a given dataset.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        criterion (nn.Module): Loss function.
    
    Returns:
        float: Average loss on the dataset.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
    
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss

def evaluate_model(model, dataloader):
    """
    Evaluate the model's accuracy and F1-score on a given dataset.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
    
    Returns:
        float: Accuracy of the model on the dataset.
        float: F1-score of the model on the dataset.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Convert lists to tensors
    all_labels = torch.tensor(all_labels)
    all_predictions = torch.tensor(all_predictions)

    # Calculate accuracy
    accuracy = torch.mean((all_predictions == all_labels).float()).item()

    # Calculate F1-score
    num_classes = len(torch.unique(all_labels))
    f1_scores = []
    for cls in range(num_classes):
        true_positive = ((all_labels == cls) & (all_predictions == cls)).sum().item()
        false_positive = ((all_labels != cls) & (all_predictions == cls)).sum().item()
        false_negative = ((all_labels == cls) & (all_predictions != cls)).sum().item()

        precision = true_positive / (true_positive + false_positive + 1e-10)  # Avoid division by zero
        recall = true_positive / (true_positive + false_negative + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        f1_scores.append(f1)

    f1 = np.mean(f1_scores)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return accuracy, f1
