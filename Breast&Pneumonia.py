# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
import medmnist
from medmnist import INFO, Evaluator
from medmnist import BreastMNIST, PneumoniaMNIST
import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import random
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Set random seed and device
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preprocessing and loading functions
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_dataset(dataset_name, sample_size=100):  # Reduced to 100 samples
    if dataset_name == 'breast':
        dataset = BreastMNIST
    elif dataset_name == 'pneumonia':
        dataset = PneumoniaMNIST
    
    data_transform = get_transform()
    
    # Load datasets
    train_dataset = dataset(split='train', transform=data_transform, download=True)
    val_dataset = dataset(split='val', transform=data_transform, download=True)
    test_dataset = dataset(split='test', transform=data_transform, download=True)
    
    # Reduce dataset sizes
    train_indices = random.sample(range(len(train_dataset)), min(sample_size, len(train_dataset)))
    val_indices = random.sample(range(len(val_dataset)), min(sample_size//2, len(val_dataset)))
    test_indices = random.sample(range(len(test_dataset)), min(sample_size//2, len(test_dataset)))
    
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Model definitions
class ResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Model, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)

class DenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.densenet.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        return self.densenet(x)

# Training and evaluation functions
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    best_val_auc = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long().squeeze()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.squeeze()
                outputs = model(inputs)
                val_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Calculate AUC
        try:
            val_auc = roc_auc_score(val_targets, val_preds[:, 1])
        except ValueError:
            val_auc = roc_auc_score(np.eye(outputs.shape[1])[val_targets], val_preds, multi_class='ovr')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}')
    
    return best_model

def test_model(model, test_loader):
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.squeeze()
            outputs = model(inputs)
            test_preds.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            test_targets.extend(targets.numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    try:
        test_auc = roc_auc_score(test_targets, test_preds[:, 1])
    except ValueError:
        test_auc = roc_auc_score(np.eye(outputs.shape[1])[test_targets], test_preds, multi_class='ovr')
    
    return test_auc

# Training loop for all models and datasets
datasets = ['breast', 'pneumonia']
results = {}

# ResNet18 for all datasets
print("\nTraining ResNet18 for all datasets...")
for dataset_name in datasets:
    print(f"\nProcessing {dataset_name} dataset")
    train_loader, val_loader, test_loader = load_dataset(dataset_name)
    num_classes = 2
    
    model = ResNet18Model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_model = train_model(model, train_loader, val_loader, criterion, optimizer)
    model.load_state_dict(best_model)
    test_auc = test_model(model, test_loader)
    
    results[f'ResNet18_{dataset_name}'] = test_auc
    torch.save(best_model, f'ResNet18_{dataset_name}_best.pth')
    print(f"Test AUC for {dataset_name}: {test_auc:.4f}")

# ViT for all datasets
print("\nTraining ViT for all datasets...")
for dataset_name in datasets:
    print(f"\nProcessing {dataset_name} dataset")
    train_loader, val_loader, test_loader = load_dataset(dataset_name)
    num_classes = 2
    
    model = ViTModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_model = train_model(model, train_loader, val_loader, criterion, optimizer)
    model.load_state_dict(best_model)
    test_auc = test_model(model, test_loader)
    
    results[f'ViT_{dataset_name}'] = test_auc
    torch.save(best_model, f'ViT_{dataset_name}_best.pth')
    print(f"Test AUC for {dataset_name}: {test_auc:.4f}")

# DenseNet for all datasets
print("\nTraining DenseNet for all datasets...")
for dataset_name in datasets:
    print(f"\nProcessing {dataset_name} dataset")
    train_loader, val_loader, test_loader = load_dataset(dataset_name)
    num_classes = 2
    
    model = DenseNetModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_model = train_model(model, train_loader, val_loader, criterion, optimizer)
    model.load_state_dict(best_model)
    test_auc = test_model(model, test_loader)
    
    results[f'DenseNet_{dataset_name}'] = test_auc
    torch.save(best_model, f'DenseNet_{dataset_name}_best.pth')
    print(f"Test AUC for {dataset_name}: {test_auc:.4f}")

# Print final results
print("\nFinal Results:")
for model_dataset, auc in results.items():
    print(f"{model_dataset}: {auc:.4f}")

# Find best model for each dataset
for dataset_name in datasets:
    dataset_results = {k: v for k, v in results.items() if dataset_name in k}
    best_model = max(dataset_results.items(), key=lambda x: x[1])
    print(f"\nBest model for {dataset_name}: {best_model[0]} with AUC: {best_model[1]:.4f}")