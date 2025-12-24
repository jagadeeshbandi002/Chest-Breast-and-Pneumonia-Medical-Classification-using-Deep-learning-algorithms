# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
import medmnist
from medmnist import INFO, Evaluator
from medmnist import ChestMNIST
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

def load_dataset(dataset_name, sample_size=1000):  # Increased sample size
    if dataset_name == 'chest':
        dataset = ChestMNIST
    
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
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)  # Increased batch size
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

# [Model definitions remain the same]
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

def calculate_auc(targets, preds, multi_label=True):
    try:
        if multi_label:
            # Handle cases where a class might be missing
            mask = targets.sum(axis=0) > 0
            if mask.sum() == 0:
                return 0.0
            return roc_auc_score(targets[:, mask], preds[:, mask], average='macro')
        else:
            return roc_auc_score(targets, preds, multi_class='ovr')
    except ValueError:
        return 0.0  # Return 0 if AUC cannot be calculated

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    best_val_auc = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float()  # Convert to float for BCE loss
            
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
                outputs = model(inputs)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(targets.numpy())
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        val_auc = calculate_auc(val_targets, val_preds)
        
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
            outputs = model(inputs)
            test_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            test_targets.extend(targets.numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    
    return calculate_auc(test_targets, test_preds)

# Training loop
datasets = ['chest']
results = {}

# Training for each model type
models_to_train = [
    ('ResNet18', ResNet18Model),
    ('ViT', ViTModel),
    ('DenseNet', DenseNetModel)
]

for model_name, model_class in models_to_train:
    print(f"\nTraining {model_name} for chest dataset...")
    train_loader, val_loader, test_loader = load_dataset('chest')
    num_classes = 14
    
    model = model_class(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_model = train_model(model, train_loader, val_loader, criterion, optimizer)
    model.load_state_dict(best_model)
    test_auc = test_model(model, test_loader)
    
    results[f'{model_name}_chest'] = test_auc
    torch.save(best_model, f'{model_name}_chest_best.pth')
    print(f"Test AUC for chest: {test_auc:.4f}")

# Print final results
print("\nFinal Results:")
for model_dataset, auc in results.items():
    print(f"{model_dataset}: {auc:.4f}")

# Find best model
best_model = max(results.items(), key=lambda x: x[1])
print(f"\nBest model overall: {best_model[0]} with AUC: {best_model[1]:.4f}")