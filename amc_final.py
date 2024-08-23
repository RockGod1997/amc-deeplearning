
import numpy as np
import pandas as pd
import h5py
import json
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
# Load HDF5 and JSON files containing the dataset and modulation classes
hdf5_file = h5py.File("", 'r')  #place the RadioML2018.01A HDF5 file path here 
modulation_classes = json.load(open("classes-fixed.json", 'r'))
iq_data = hdf5_file['X']  # IQ samples
modulation_labels = np.argmax(hdf5_file['Y'], axis=1)  # Modulation classes
snr_values = hdf5_file['Z'][:, 0]  # Signal-to-noise ratios

# Constants
N_CHANNELS = 2
BATCH_SIZE = 32
N_FRAMES_TRAIN = 1024
N_FRAMES_VALID = 512
N_FRAMES_TEST = 256

# Function to split dataset into train, validation, and test sets
def dataset_split(iq_samples, modulation_classes, modulation_labels, snr_values, target_modulations, mode, target_snrs, 
                  train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, seed=48):
    """
    Splits the dataset into train, validation, and test sets based on the specified proportions.
    """
    np.random.seed(seed)
    total_samples = 4096  # The number of samples per modulation/SNR combination
    train_idx = int(train_ratio * total_samples)
    valid_idx = int((train_ratio + valid_ratio) * total_samples)
    test_idx = int((train_ratio + valid_ratio + test_ratio) * total_samples)
    
    X_split, Y_split, Z_split = [], [], []
    
    target_modulation_indices = [modulation_classes.index(mod) for mod in target_modulations]
    
    for mod in target_modulation_indices:
        for snr in target_snrs:
            indices = np.where((modulation_labels == mod) & (snr_values == snr))[0]
            np.random.shuffle(indices)
            train, valid, test, _ = np.split(indices, [train_idx, valid_idx, test_idx])
            
            if mode == 'train':
                X_split.append(iq_samples[train])
                Y_split.append(modulation_labels[train])
                Z_split.append(snr_values[train])
            elif mode == 'valid':
                X_split.append(iq_samples[valid])
                Y_split.append(modulation_labels[valid])
                Z_split.append(snr_values[valid])
            elif mode == 'test':
                X_split.append(iq_samples[test])
                Y_split.append(modulation_labels[test])
                Z_split.append(snr_values[test])
            else:
                raise ValueError(f"Unknown mode: {mode}. Valid modes are 'train', 'valid', and 'test'")
    
    X_output = np.vstack(X_split)
    Y_output = np.concatenate(Y_split)
    Z_output = np.concatenate(Z_split)
    
    # Re-index labels to be continuous starting from 0
    for idx, label in enumerate(np.unique(Y_output)):
        Y_output[Y_output == label] = idx
        
    return X_output, Y_output, Z_output

# Custom dataset class for RadioML2018.01A
class DeepSigDataset(Dataset):
    def __init__(self, mode: str, seed=48):
        """
        Initializes the RadioML18 dataset for a specified mode (train, valid, test).
        """
        super().__init__()
        hdf5_file = h5py.File("/kaggle/input/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5", 'r')
        self.modulation_classes = json.load(open("/kaggle/input/radioml2018/classes-fixed.json", 'r'))
        self.iq_samples = hdf5_file['X']
        self.modulation_labels = np.argmax(hdf5_file['Y'], axis=1)
        self.snr_values = hdf5_file['Z'][:, 0]
        
        # Calculate the data split proportions
        total_samples = self.iq_samples.shape[0]
        train_ratio = (24 * 26 * N_FRAMES_TRAIN) / total_samples
        valid_ratio = (24 * 26 * N_FRAMES_VALID) / total_samples
        test_ratio = (24 * 26 * N_FRAMES_TEST) / total_samples
        
        # Target modulation classes and SNRs
        self.target_modulations = ['OOK', '4ASK', 'BPSK']
        self.target_snrs = np.unique(self.snr_values)
        
        # Split the data into train, valid, or test
        self.X_data, self.Y_data, self.Z_data = dataset_split(
            iq_samples=self.iq_samples,
            modulation_classes=self.modulation_classes,
            modulation_labels=self.modulation_labels,
            snr_values=self.snr_values,
            mode=mode,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            target_modulations=self.target_modulations,
            target_snrs=self.target_snrs,
            seed=seed
        )
        
        # Dataset statistics
        self.num_samples = self.X_data.shape[0]
        self.num_labels = len(self.target_modulations)
        self.num_snrs = len(self.target_snrs)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        iq_sample = torch.Tensor(self.X_data[idx]).transpose(0, 1)
        modulation_label = self.Y_data[idx]
        snr_value = self.Z_data[idx]
        return iq_sample, modulation_label, snr_value

# Initialize dataset loaders for training, validation, and testing
train_loader = DataLoader(dataset=DeepSigDataset(mode='train'), batch_size=64, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset=DeepSigDataset(mode='valid'), batch_size=128, shuffle=False, drop_last=False)
test_loader = DataLoader(dataset=DeepSigDataset(mode='test'), batch_size=128, shuffle=False, drop_last=False)

# Define CNN block for convolutional layers
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
    def forward(self, x):
        return self.layer(x)

# Define CNN-based neural network model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            CNNBlock(2, 24),
            CNNBlock(24, 24),
            CNNBlock(24, 48),
            CNNBlock(48, 48),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 modulation classes
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# Print model summary
model = CNNModel()
summary(model.to('cpu'), (BATCH_SIZE, N_CHANNELS, 1024))

#LSTM Model
class LSTMModel(nn.Module):
    def __init__(self,):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(2, 128, 3, batch_first=True) 


        self.fc = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),  
            nn.Linear(128,3),
        )
        
    def forward(self, x):

        out, (hn, cn) = self.lstm(x.transpose(1,2))

        out = self.fc(hn[-1])
        return out


summary(LSTMModel().to('cpu'), (BATCH_SIZE,N_CHANNELS,1024))

# GRU Model
class GRUModel(nn.Module):
    def __init__(self,):
        super(GRUModel, self).__init__()
        
        self.GRU = nn.GRU(2, 128, 4)

        self.fc = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),  
            nn.Linear(128,3),
        )
        
    def forward(self, x):
        out, hn = self.GRU(x.transpose(1,2))
        out = self.fc(out[:,-1,:])
        return out
    
summary(GRUModel().to('cpu'), (BATCH_SIZE,N_CHANNELS,1024))

# Transformer Model
class TransformerModel(nn.Module):

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.model_type = 'Transformer'
        self.conv_backbone = nn.Sequential(
           nn.Conv1d(2,32,4,4),
           nn.BatchNorm1d(32),
           nn.LeakyReLU(),
           nn.Conv1d(32,64,4,4),
           nn.BatchNorm1d(64),
           nn.LeakyReLU(), 

        )
        self.pos_encoder = PositionalEncoding(64, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=64,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 5)
        self.d_model =64
        self.linear = nn.Sequential(
        nn.Linear(64, 32),
        nn.Linear(32,3)
        )

    def forward(self, src, src_mask= None):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.conv_backbone(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = torch.mean(output,dim=1)
        output = self.linear(output)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
summary(TransformerModel().to('cpu'), (BATCH_SIZE,N_CHANNELS,1024))
# Training function
def train_model(model, train_loader, valid_loader, num_epochs=30, device='cuda'):
    """
    Trains the model and validates it at each epoch.
    """
    model.to(device)
    
    # Initialize lists to track metrics
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    # Optimizer, scheduler, and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in trange(num_epochs, unit='epoch'):
        # Training phase
        model.train()
        total_train_loss, total_train_accuracy = 0, 0
        
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_accuracy += (outputs.argmax(dim=-1) == labels).float().mean().item()
        
        # Validation phase
        model.eval()
        total_val_loss, total_val_accuracy = 0, 0
        
        with torch.no_grad():
            for inputs, labels, _ in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                total_val_accuracy += (outputs.argmax(dim=-1) == labels).float().mean().item()
        
        # Scheduler step
        scheduler.step()
        
        # Calculate average losses and accuracies for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        avg_val_loss = total_val_loss / len(valid_loader)
        avg_val_accuracy = total_val_accuracy / len(valid_loader)
        
        # Append metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
    
    # Return training history
    return {'train_loss': train_losses, 'train_accuracy': train_accuracies,
            'val_loss': val_losses, 'val_accuracy': val_accuracies}

# Testing function
def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluates the model on the test set.
    """
    model.to(device)
    model.eval()
    
    all_predictions, all_labels, all_snrs = [], [], []
    
    with torch.no_grad():
        for inputs, labels, snrs in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_snrs.append(snrs.cpu().numpy())
    
    return np.concatenate(all_predictions), np.concatenate(all_labels), np.concatenate(all_snrs)

# Plot training history
def plot_training_history(history):
    """
    Plots the training and validation loss and accuracy curves.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

# Plot accuracy per modulation and SNR
def plot_test_accuracy(predictions, labels, snrs, modulation_classes, snr_values):
    """
    Plots test accuracy per modulation type and SNR.
    """
    accuracy_df = pd.DataFrame({'predictions': predictions, 'labels': labels, 'snrs': snrs})
    
    accuracies = []
    for mod_class in range(len(modulation_classes)):
        for snr in snr_values:
            relevant_samples = accuracy_df[(accuracy_df['labels'] == mod_class) & (accuracy_df['snrs'] == snr)]
            accuracy = np.mean(relevant_samples['predictions'] == relevant_samples['labels'])
            accuracies.append((mod_class, snr, accuracy))
    
    accuracy_df = pd.DataFrame(accuracies, columns=['mod_class', 'snr', 'accuracy'])
    
    plt.figure(figsize=(12, 6))
    for mod_class in range(len(modulation_classes)):
        mod_accuracies = accuracy_df[accuracy_df['mod_class'] == mod_class]
        plt.plot(mod_accuracies['snr'], mod_accuracies['accuracy'], label=modulation_classes[mod_class])
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title("Test Accuracy per Modulation Type and SNR")
    plt.show()

# Training and evaluation pipeline
def train_and_evaluate_model(model, train_loader, valid_loader, test_loader, num_epochs=50, device='cuda'):
    """
    Trains the model, evaluates it, and plots the results.
    """
    # Train the model and capture the training history
    training_history = train_model(model, train_loader, valid_loader, num_epochs=num_epochs, device=device)
    
    # Plot training history
    plot_training_history(training_history)
    
    # Evaluate the model on the test set
    predictions, labels, snrs = evaluate_model(model, test_loader, device=device)
    
    # Plot test accuracy per modulation type and SNR
    plot_test_accuracy(predictions, labels, snrs, modulation_classes=['OOK', '4ASK', 'BPSK'], snr_values=np.unique(snrs))
    
    return model

# Run the training and evaluation process for the  models
cnn_model = CNNModel()
gru_model=GRUModel()
lstm_model= LSTMModel()
transformer_model=TransformerModel()

trained_cnn_model = train_and_evaluate_model(cnn_model, train_loader, valid_loader, test_loader, num_epochs=50, device='cpu')
trained_lstm_model = train_and_evaluate_model(lstm_model, train_loader, valid_loader, test_loader, num_epochs=50, device='cpu')
trained_gru_model = train_and_evaluate_model(gru_model, train_loader, valid_loader, test_loader, num_epochs=50, device='cpu')
trained_transformer_model = train_and_evaluate_model(transformer_model, train_loader, valid_loader, test_loader, num_epochs=50, device='cpu')

