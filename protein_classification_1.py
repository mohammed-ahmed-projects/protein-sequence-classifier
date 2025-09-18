#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein Sequence Classification - Complete Project

This project includes:
- Data preprocessing
- Sequence to numerical representation conversion
- Building different models (CNN, LSTM, Transformer)
- Training and evaluation
- Creating visualizations and results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Alternative models will be used.")

# Machine Learning Libraries
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Alternative functions will be implemented.")

# BioPython for FASTA handling
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. Alternative FASTA parser will be used.")

class ProteinDataProcessor:
    """
    Protein Data Processor
    """
    
    def __init__(self):
        # Standard amino acids list
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {idx: aa for idx, aa in enumerate(self.amino_acids)}
        
    def clean_sequence(self, sequence):
        """
        Clean protein sequence from unusual characters
        """
        sequence = sequence.upper().strip()
        cleaned = ''.join([aa for aa in sequence if aa in self.amino_acids])
        return cleaned
    
    def sequence_to_onehot(self, sequence, max_length=None):
        """
        Convert sequence to One-hot encoding
        """
        if max_length is None:
            max_length = len(sequence)
        
        onehot = np.zeros((max_length, len(self.amino_acids)))
        
        for i, aa in enumerate(sequence[:max_length]):
            if aa in self.aa_to_idx:
                onehot[i, self.aa_to_idx[aa]] = 1
                
        return onehot
    
    def sequence_to_indices(self, sequence, max_length=None):
        """
        Convert sequence to numerical indices
        """
        if max_length is None:
            max_length = len(sequence)
            
        indices = []
        for aa in sequence[:max_length]:
            if aa in self.aa_to_idx:
                indices.append(self.aa_to_idx[aa])
            else:
                indices.append(0)
                
        while len(indices) < max_length:
            indices.append(0)
            
        return np.array(indices)
    
    def load_fasta_file(self, filepath):
        """
        Load FASTA file
        """
        sequences = []
        labels = []
        
        if BIOPYTHON_AVAILABLE:
            for record in SeqIO.parse(filepath, "fasta"):
                sequence = str(record.seq)
                label = 1 if 'enzyme' in record.description.lower() else 0
                sequences.append(self.clean_sequence(sequence))
                labels.append(label)
        else:
            with open(filepath, 'r') as f:
                content = f.read()
                records = content.split('>')
                for record in records[1:]:
                    lines = record.strip().split('\n')
                    if len(lines) >= 2:
                        description = lines[0]
                        sequence = ''.join(lines[1:])
                        label = 1 if 'enzyme' in description.lower() else 0
                        sequences.append(self.clean_sequence(sequence))
                        labels.append(label)
        
        return sequences, labels
    
    def create_sample_dataset(self, n_samples=1000):
        """
        Create sample dataset
        """
        np.random.seed(42)
        sequences = []
        labels = []
        
        for i in range(n_samples):
            length = np.random.randint(50, 500)
            sequence = ''.join(np.random.choice(self.amino_acids, length))
            
            enzyme_indicators = ['E', 'D', 'H', 'C']
            enzyme_count = sum([sequence.count(aa) for aa in enzyme_indicators])
            label = 1 if enzyme_count > length * 0.15 else 0
            
            sequences.append(sequence)
            labels.append(label)
        
        return sequences, labels

class ProteinDataset(Dataset):
    """
    Protein Dataset for PyTorch
    """
    
    def __init__(self, sequences, labels, processor, max_length=500, encoding='onehot'):
        self.sequences = sequences
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
        self.encoding = encoding
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.encoding == 'onehot':
            encoded = self.processor.sequence_to_onehot(sequence, self.max_length)
            encoded = torch.FloatTensor(encoded).transpose(0, 1)
        else:
            encoded = self.processor.sequence_to_indices(sequence, self.max_length)
            encoded = torch.LongTensor(encoded)
        
        return encoded, torch.LongTensor([label])

class ProteinCNN(nn.Module):
    """
    CNN Model for Protein Classification
    """
    
    def __init__(self, input_size=20, num_classes=2):
        super(ProteinCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = torch.mean(x, dim=2)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class ProteinLSTM(nn.Module):
    """
    LSTM Model for Protein Classification
    """
    
    def __init__(self, vocab_size=20, embedding_dim=64, hidden_dim=128, num_classes=2):
        super(ProteinLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        output = lstm_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

class ProteinClassifier:
    """
    Main Protein Classifier
    """
    
    def __init__(self):
        self.processor = ProteinDataProcessor()
        self.models = {}
        self.results = {}
        
    def prepare_data(self, sequences, labels, test_size=0.2, val_size=0.1):
        """
        Prepare data for training
        """
        if SKLEARN_AVAILABLE:
            X_temp, X_test, y_temp, y_test = train_test_split(
                sequences, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
            )
        else:
            n_total = len(sequences)
            n_test = int(n_total * test_size)
            n_val = int(n_total * val_size)
            
            indices = np.random.permutation(n_total)
            test_idx = indices[:n_test]
            val_idx = indices[n_test:n_test+n_val]
            train_idx = indices[n_test+n_val:]
            
            X_train = [sequences[i] for i in train_idx]
            X_val = [sequences[i] for i in val_idx]
            X_test = [sequences[i] for i in test_idx]
            y_train = [labels[i] for i in train_idx]
            y_val = [labels[i] for i in val_idx]
            y_test = [labels[i] for i in test_idx]
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def train_pytorch_model(self, model, train_loader, val_loader, epochs=10, lr=0.001):
        """
        Train PyTorch model
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Loss: {val_losses[-1]:.4f}, '
                  f'Val Acc: {val_acc:.2f}%')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    def evaluate_model(self, model, test_loader):  
    """  
    Evaluate model  
    """  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model.to(device)  
    model.eval()  
    
    all_predictions = []  
    all_labels = []  
    all_probabilities = []  
    
    with torch.no_grad():  
        for batch_x, batch_y in test_loader:  
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()  
            outputs = model(batch_x)  
            probabilities = torch.softmax(outputs, dim=1)  
            _, predicted = torch.max(outputs, 1)  
            
            all_predictions.extend(predicted.cpu().numpy())  
            all_labels.extend(batch_y.cpu().numpy())  
            all_probabilities.extend(probabilities.cpu().numpy())  
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)  

def calculate_metrics(self, y_true, y_pred, y_prob=None):  
    """  
    Calculate metrics  
    """  
    if SKLEARN_AVAILABLE:  
        accuracy = accuracy_score(y_true, y_pred)  
        precision = precision_score(y_true, y_pred, average='weighted')  
        recall = recall_score(y_true, y_pred, average='weighted')  
        f1 = f1_score(y_true, y_pred, average='weighted')  
        
        metrics = {  
            'accuracy': accuracy,  
            'precision': precision,  
            'recall': recall,  
            'f1_score': f1  
        }  
        
        if y_prob is not None and len(np.unique(y_true)) == 2:  
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])  
            auc_score = auc(fpr, tpr)  
            metrics['auc'] = auc_score  
            metrics['fpr'] = fpr  
            metrics['tpr'] = tpr  
    else:  
        accuracy = np.mean(y_true == y_pred)  
        
        tp = np.sum((y_true == 1) & (y_pred == 1))  
        fp = np.sum((y_true == 0) & (y_pred == 1))  
        fn = np.sum((y_true == 1) & (y_pred == 0))  
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  
        
        metrics = {  
            'accuracy': accuracy,  
            'precision': precision,  
            'recall': recall,  
            'f1_score': f1  
        }  
    
    return metrics  

def plot_training_history(self, history, model_name):  
    """  
    Plot training history  
    """  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  
    
    # Loss plot  
    ax1.plot(history['train_losses'], label='Train Loss')  
    ax1.plot(history['val_losses'], label='Validation Loss')  
    ax1.set_title(f'{model_name} - Training Loss')  
    ax1.set_xlabel('Epoch')  
    ax1.set_ylabel('Loss')  
    ax1.legend()  
    ax1.grid(True)  
    
    # Accuracy plot  
    ax2.plot(history['train_accs'], label='Train Accuracy')  
    ax2.plot(history['val_accs'], label='Validation Accuracy')  
    ax2.set_title(f'{model_name} - Training Accuracy')  
    ax2.set_xlabel('Epoch')  
    ax2.set_ylabel('Accuracy (%)')  
    ax2.legend()  
    ax2.grid(True)  
    
    plt.tight_layout()  
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')  
    plt.show()  

def plot_confusion_matrix(self, y_true, y_pred, model_name):  
    """  
    Plot confusion matrix  
    """  
    if SKLEARN_AVAILABLE:  
        cm = confusion_matrix(y_true, y_pred)  
    else:  
        cm = np.zeros((2, 2))  
        for true, pred in zip(y_true, y_pred):  
            cm[true, pred] += 1  
    
    plt.figure(figsize=(8, 6))  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',   
                xticklabels=['Non-Enzyme', 'Enzyme'],  
                yticklabels=['Non-Enzyme', 'Enzyme'])  
    plt.title(f'{model_name} - Confusion Matrix')  
    plt.xlabel('Predicted')  
    plt.ylabel('Actual')  
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')  
    plt.show()  

def plot_roc_curve(self, metrics, model_name):  
    """  
    Plot ROC curve  
    """  
    if 'fpr' in metrics and 'tpr' in metrics:  
        plt.figure(figsize=(8, 6))  
        plt.plot(metrics['fpr'], metrics['tpr'],   
                label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')  
        plt.plot([0, 1], [0, 1], 'k--', label='Random')  
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate')  
        plt.title(f'{model_name} - ROC Curve')  
        plt.legend()  
        plt.grid(True)  
        plt.savefig(f'{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')  
        plt.show()  

def analyze_sequences(self, sequences, labels):  
    """  
    Analyze sequences  
    """  
    print("=== Data Analysis ===")  
    
    lengths = [len(seq) for seq in sequences]  
    enzyme_count = sum(labels)  
    non_enzyme_count = len(labels) - enzyme_count  
    
    print(f"Total sequences: {len(sequences)}")  
    print(f"Enzymes: {enzyme_count} ({enzyme_count/len(labels)*100:.1f}%)")  
    print(f"Non-enzymes: {non_enzyme_count} ({non_enzyme_count/len(labels)*100:.1f}%)")  
    print(f"Average sequence length: {np.mean(lengths):.1f}")  
    print(f"Shortest sequence: {min(lengths)}")  
    print(f"Longest sequence: {max(lengths)}")  
    
    # Distribution plots  
    plt.figure(figsize=(12, 4))  
    
    plt.subplot(1, 2, 1)  
    plt.hist(lengths, bins=50, alpha=0.7, edgecolor='black')  
    plt.xlabel('Sequence Length')  
    plt.ylabel('Frequency')  
    plt.title('Distribution of Sequence Lengths')  
    plt.grid(True, alpha=0.3)  
    
    plt.subplot(1, 2, 2)  
    labels_names = ['Non-Enzyme', 'Enzyme']  
    counts = [non_enzyme_count, enzyme_count]  
    plt.pie(counts, labels=labels_names, autopct='%1.1f%%', startangle=90)  
    plt.title('Class Distribution')  
    
    plt.tight_layout()  
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')  
    plt.show()  
    
    # Amino acid composition analysis  
    enzyme_seqs = [sequences[i] for i in range(len(sequences)) if labels[i] == 1]  
    non_enzyme_seqs = [sequences[i] for i in range(len(sequences)) if labels[i] == 0]  
    
    enzyme_aa_freq = Counter(''.join(enzyme_seqs))  
    non_enzyme_aa_freq = Counter(''.join(non_enzyme_seqs))  
    
    enzyme_total = sum(enzyme_aa_freq.values())  
    non_enzyme_total = sum(non_enzyme_aa_freq.values())  
    
    aa_comparison = []  
    for aa in self.processor.amino_acids:  
        enzyme_freq = enzyme_aa_freq.get(aa, 0) / enzyme_total * 100  
        non_enzyme_freq = non_enzyme_aa_freq.get(aa, 0) / non_enzyme_total * 100  
        aa_comparison.append({  
            'amino_acid': aa,  
            'enzyme_freq': enzyme_freq,  
            'non_enzyme_freq': non_enzyme_freq,  
            'difference': enzyme_freq - non_enzyme_freq  
        })  
    
    aa_df = pd.DataFrame(aa_comparison)  
    
    plt.figure(figsize=(15, 6))  
    x = np.arange(len(self.processor.amino_acids))  
    width = 0.35  
    
    plt.bar(x - width/2, aa_df['enzyme_freq'], width, label='Enzyme', alpha=0.8)  
    plt.bar(x + width/2, aa_df['non_enzyme_freq'], width, label='Non-Enzyme', alpha=0.8)  
    
    plt.xlabel('Amino Acids')  
    plt.ylabel('Frequency (%)')  
    plt.title('Amino Acid Frequency Comparison')  
    plt.xticks(x, self.processor.amino_acids)  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()  
    plt.savefig('amino_acid_analysis.png', dpi=300, bbox_inches='tight')  
    plt.show()  
    
    return aa_df  

def run_complete_pipeline(self, sequences=None, labels=None, use_sample_data=True):  
    """  
    Run complete pipeline  
    """  
    print("=== Starting Protein Sequence Classification Project ===")  
    
    if use_sample_data or sequences is None:  
        print("Creating sample data...")  
        sequences, labels = self.processor.create_sample_dataset(n_samples=2000)  
    
    aa_analysis = self.analyze_sequences(sequences, labels)  
    (X_train, X_val, X_test), (y_train, y_val, y_test) = self.prepare_data(sequences, labels)  
    
    print(f"\nData split:")  
    print(f"Training: {len(X_train)} samples")  
    print(f"Validation: {len(X_val)} samples")  
    print(f"Testing: {len(X_test)} samples")  
    
    results_summary = []  
    
    if TORCH_AVAILABLE:  
        # Train CNN model  
        print("\n=== Training CNN Model ===")  
        
        train_dataset = ProteinDataset(X_train, y_train, self.processor, encoding='onehot')  
        val_dataset = ProteinDataset(X_val, y_val, self.processor, encoding='onehot')  
        test_dataset = ProteinDataset(X_test, y_test, self.processor, encoding='onehot')  
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  
        
        cnn_model = ProteinCNN()  
        cnn_history = self.train_pytorch_model(cnn_model, train_loader, val_loader, epochs=20)  
        
        cnn_pred, cnn_true, cnn_prob = self.evaluate_model(cnn_model, test_loader)  
        cnn_metrics = self.calculate_metrics(cnn_true, cnn_pred, cnn_prob)  
        
        print(f"CNN Results:")  
        for metric, value in cnn_metrics.items():  
            if metric not in ['fpr', 'tpr']:  
                print(f"{metric}: {value:.4f}")  
        
        self.plot_training_history(cnn_history, 'CNN')  
        self.plot_confusion_matrix(cnn_true, cnn_pred, 'CNN')  
        if 'auc' in cnn_metrics:  
            self.plot_roc_curve(cnn_metrics, 'CNN')  
        
        results_summary.append({  
            'Model': 'CNN',  
            'Accuracy': cnn_metrics['accuracy'],  
            'Precision': cnn_metrics['precision'],  
            'Recall': cnn_metrics['recall'],  
            'F1-Score': cnn_metrics['f1_score'],  
            'AUC': cnn_metrics.get('auc', 'N/A')  
        })  
        
        # Train LSTM model  
        print("\n=== Training LSTM Model ===")  
        
        train_dataset_lstm = ProteinDataset(X_train, y_train, self.processor, encoding='indices')  
        val_dataset_lstm = ProteinDataset(X_val, y_val, self.processor, encoding='indices')  
        test_dataset_lstm = ProteinDataset(X_test, y_test, self.processor, encoding='indices')  
        
        train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=32, shuffle=True)  
        val_loader_lstm = DataLoader(val_dataset_lstm, batch_size=32, shuffle=False)  
        test_loader_lstm = DataLoader(test_dataset_lstm, batch_size=32, shuffle=False)  
        
        lstm_model = ProteinLSTM()  
        lstm_history = self.train_pytorch_model(lstm_model, train_loader_lstm, val_loader_lstm, epochs=20)  
        
        lstm_pred, lstm_true, lstm_prob = self.evaluate_model(lstm_model, test_loader_lstm)  
        lstm_metrics = self.calculate_metrics(lstm_true, lstm_pred, lstm_prob)  
        
        print(f"LSTM Results:")  
        for metric, value in lstm_metrics.items():  
            if metric not in ['fpr', 'tpr']:  
                print(f"{metric}: {value:.4f}")  
        
        self.plot_training_history(lstm_history, 'LSTM')  
        self.plot_confusion_matrix(lstm_true, lstm_pred, 'LSTM')  
        if 'auc' in lstm_metrics:  
            self.plot_roc_curve(lstm_metrics, 'LSTM')  
        
        results_summary.append({  
            'Model': 'LSTM',  
            'Accuracy': lstm_metrics['accuracy'],  
            'Precision': lstm_metrics['precision'],  
            'Recall': lstm_metrics['recall'],  
            'F1-Score': lstm_metrics['f1_score'],  
            'AUC': lstm_metrics.get('auc', 'N/A')  
        })  
        
        torch.save(cnn_model.state_dict(), 'protein_cnn_model.pth')  
        torch.save(lstm_model.state_dict(), 'protein_lstm_model.pth')  
        print("Models saved successfully")  
    
    else:  
        print("PyTorch not available. Skipping deep learning models.")  
    
    if results_summary:  
        print("\n=== Results Summary ===")  
        
        results_df = pd.DataFrame(results_summary)  
        print(results_df.to_string(index=False))  
        
        plt.figure(figsize=(12, 8))  
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']  
        x = np.arange(len(results_df))  
        width = 0.2  
        
        for i, metric in enumerate(metrics_to_plot):  
            plt.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)  
        
        plt.xlabel('Models')  
        plt.ylabel('Score')  
        plt.title('Model Performance Comparison')  
        plt.xticks(x + width*1.5, results_df['Model'])  
        plt.legend()  
        plt.grid(True, alpha=0.3)  
        plt.ylim(0, 1)  
        
        for i, model in enumerate(results_df['Model']):  
            for j, metric in enumerate(metrics_to_plot):  
                value = results_df.iloc[i][metric]  
                plt.text(i + j*width, value + 0.01, f'{value:.3f}',   
                        ha='center', va='bottom', fontsize=8)  
        
        plt.tight_layout()  
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')  
        plt.show()  
        
        results_df.to_csv('model_results.csv', index=False)  
        aa_analysis.to_csv('amino_acid_analysis.csv', index=False)  
    
    print("\n=== Project Completed Successfully ===")  
    
    return results_summary, aa_analysis  

def main():  
    """  
    Main function  
    """  
    classifier = ProteinClassifier()  
    results, analysis = classifier.run_complete_pipeline(use_sample_data=True)  
    
    print("\nSaved files:")  
    print("- model_results.csv")  
    print("- amino_acid_analysis.csv")  
    print("- data_analysis.png")  
    print("- amino_acid_analysis.png")  
    print("- model_comparison.png")  
    if TORCH_AVAILABLE:  
        print("- protein_cnn_model.pth")  
        print("- protein_lstm_model.pth")  
        print("- CNN_training_history.png")  
        print("- CNN_confusion_matrix.png")  
        print("- CNN_roc_curve.png")  
        print("- LSTM_training_history.png")  
        print("- LSTM_confusion_matrix.png")  
        print("- LSTM_roc_curve.png")  

if __name__ == "__main__":  
    main()
