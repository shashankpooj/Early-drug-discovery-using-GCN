import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

from backend.Targetmodel import HitModel

# Function to preprocess the dataset
def preprocess_hit_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    print("Columns in dataset:", df.columns)
    
    # Drop rows with missing SMILES or activity outcomes
    df = df.dropna(subset=['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME'])
    
    # Convert activity outcome to binary label
    df['PUBCHEM_ACTIVITY_OUTCOME'] = df['PUBCHEM_ACTIVITY_OUTCOME'].map({'Active': 1, 'Inactive': 0})
    
    # Extract SMILES strings and labels
    smiles = df['PUBCHEM_EXT_DATASOURCE_SMILES']
    labels = df['PUBCHEM_ACTIVITY_OUTCOME'].astype(int)
    
    # Generate molecular fingerprints using RDKit
    fingerprints = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = torch.zeros((1, 2048), dtype=torch.float)
            DataStructs.ConvertToNumpyArray(fp, arr.numpy())
            fingerprints.append(arr)
        else:
            fingerprints.append(torch.zeros((1, 2048), dtype=torch.float))
    
    # Combine fingerprints and labels into PyTorch Geometric Data objects
    data_list = []
    for i, fp in enumerate(fingerprints):
        x = fp  # Features are the fingerprints
        
        # Define edge_index as a fully connected graph (simple dummy graph)
        num_nodes = x.shape[0]
        
        if num_nodes > 1:
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()  # Create dummy fully connected graph
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)  # Empty edge_index for single node case
        
        y = torch.tensor(labels.iloc[i], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # If the graph is not empty, ensure edge_index is valid
        if edge_index.numel() > 0:
            assert edge_index.max() < x.shape[0], "Edge index contains out-of-bound node indices"
        
        data_list.append(data)
    
    return data_list


# Preprocess the data
file_path = './data/1.csv'  # Adjust the file path as needed
data_list = preprocess_hit_data(file_path)

# Split into training and testing datasets
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Training function
def train_hit_model(model, train_loader, test_loader, epochs, learning_rate):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output.view(-1, 2), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            pred = output.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")

# Model configuration
input_dim = 2048  # Fingerprint size
hidden_dim = 128
epochs = 50
learning_rate = 0.001

# Initialize and train the model
hit_model = HitModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=2)
train_hit_model(hit_model, train_loader, test_loader, epochs, learning_rate)

# Save the trained model
torch.save(hit_model.state_dict(), './data/saved_models/hit_model.pth')
print("Hit model saved to hit_model.pth")
