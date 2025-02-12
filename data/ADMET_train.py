import deepchem as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch.optim import Adam

# Load Tox21 dataset from DeepChem
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

# Define the ADMETModel
class ADMETModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=5):  # 5 ADMET properties
        super(ADMETModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

# Convert DeepChem dataset into PyTorch Geometric dataset
def deepchem_to_pyg(dc_dataset):
    pyg_data_list = []
    for mol in dc_dataset.iterbatches(batch_size=1, deterministic=True):
        mol = mol[0]
        smiles, features, adjacency = mol['smiles'], mol['node_features'], mol['edge_index']
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(adjacency.T, dtype=torch.long)  # Convert edge list to PyTorch format
        y = torch.tensor(mol['targets'], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        pyg_data_list.append(data)
    return pyg_data_list

train_pyg_data = deepchem_to_pyg(train_dataset)
valid_pyg_data = deepchem_to_pyg(valid_dataset)
test_pyg_data = deepchem_to_pyg(test_dataset)

# Create PyTorch Geometric data loaders
batch_size = 32
train_loader = DataLoader(train_pyg_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_pyg_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_pyg_data, batch_size=batch_size, shuffle=False)

# Training loop for ADMETModel
def train_admet_model(model, train_loader, valid_loader, epochs, learning_rate):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()  # Suitable for multi-task classification
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in valid_loader:
            output = model(batch)
            pred = torch.sigmoid(output) > 0.5  # Convert logits to binary predictions
            correct += (pred == batch.y).sum().item()
            total += batch.y.numel()
    print(f"Validation Accuracy: {correct / total:.4f}")

# Model configuration
input_dim = 75  # Default number of atom features in GraphConv featurizer
hidden_dim = 128
output_dim = len(tasks)  # Number of ADMET prediction tasks
epochs = 10
learning_rate = 0.001

# Initialize and train the model
admet_model = ADMETModel(input_dim, hidden_dim, output_dim)
train_admet_model(admet_model, train_loader, valid_loader, epochs, learning_rate)

# Save the trained model for further use
model_path = "./data/saved_models"
torch.save(admet_model.state_dict(), model_path)
print(f"Model saved to {model_path}")
