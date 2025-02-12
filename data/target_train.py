import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn import functional as F
from torch.optim import Adam

# Import the model
from backend.Targetmodel import TargetModel  # Adjust the import path as needed

# Function to extract k-mer counts
def kmer_count(sequence, k=3):
    """Extract k-mer counts from a sequence."""
    return Counter([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

def preprocess_data(positive_file, negative_file, k=3):
    """Preprocess and combine positive and negative datasets."""
    # Load datasets
    positive_df = pd.read_csv(positive_file)
    negative_df = pd.read_csv(negative_file)
    
    # Combine datasets and add labels
    positive_df['label'] = 1
    negative_df['label'] = 0
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Extract k-mer features
    all_kmers = set()
    for seq in pd.concat([combined_df['protein_sequences_1'], combined_df['protein_sequences_2']]):
        all_kmers.update(kmer_count(seq, k).keys())
    all_kmers = sorted(all_kmers)  # Ensure consistent order

    with open('./data/kmer_vocab.txt', 'w') as f:
        for kmer in all_kmers:
            f.write(f"{kmer}\n")
    
    kmer_to_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    feature_dim = len(all_kmers)
    
    def encode_sequence(sequence):
        """Encode a sequence into a feature vector using k-mer counts."""
        kmer_counts = kmer_count(sequence, k)
        features = [0] * feature_dim
        for kmer, count in kmer_counts.items():
            if kmer in kmer_to_index:
                features[kmer_to_index[kmer]] = count
        return features
    
    # Encode sequences
    combined_df['features_1'] = combined_df['protein_sequences_1'].apply(encode_sequence)
    combined_df['features_2'] = combined_df['protein_sequences_2'].apply(encode_sequence)
    
    # Combine features and labels into PyTorch Geometric Data objects
    data_list = []
    for _, row in combined_df.iterrows():
        features = torch.tensor(row['features_1'] + row['features_2'], dtype=torch.float)
        label = torch.tensor(row['label'], dtype=torch.long)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Simple edge index
        data = Data(x=features.view(2, -1), edge_index=edge_index, y=label)
        data_list.append(data)
    
    return data_list

# Load and preprocess the data
positive_file = './data/positive_protein_sequences.csv'
negative_file = './data/negative_protein_sequences.csv'
data_list = preprocess_data(positive_file, negative_file, k=3)

# Split into training and testing datasets
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train_data)}")
print(f"Number of test samples: {len(test_data)}")

# Training function
def train_model(model, train_loader, test_loader, epochs, learning_rate):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(train_loader):
            print(f"Processing batch {i + 1}")
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Debug batch processing
            if i >= 10:  # Limit to 10 batches per epoch for debugging
                print("Stopping after 10 batches (debug mode)")
                break
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
input_dim = len(data_list[0].x[0])  # Feature size from encoded sequences
print(input_dim)  # Check feature size for debugging
hidden_dim = 128
epochs = 50
learning_rate = 0.001

# Initialize and train the model
model = TargetModel(input_dim=input_dim, hidden_dim=hidden_dim)
train_model(model, train_loader, test_loader, epochs, learning_rate)

# Save the trained model
torch.save(model.state_dict(), './data/saved_models/target_model.pth')
print("Model saved to target_model.pth")

# Loading the k-mer vocabulary (ensure consistent feature encoding during prediction)
kmer_vocab_file = './data/kmer_vocab.txt'

with open(kmer_vocab_file, 'r') as file:
    all_kmers = [line.strip() for line in file.readlines()]

kmer_to_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
feature_dim = len(all_kmers)

# Function to predict the label for a given protein sequence and its confidence
def predict_protein_sequence(sequence, model, k=3):
    # Encode the input sequence using the same k-mer vocabulary
    kmer_counts = kmer_count(sequence, k)
    
    features = [0] * feature_dim
    for kmer, count in kmer_counts.items():
        if kmer in kmer_to_index:
            features[kmer_to_index[kmer]] = count

    # Create a Data object for the sequence
    features_tensor = torch.tensor(features, dtype=torch.float).view(1, -1)
    
    # For a single sequence, there are no edges, so edge_index is an empty tensor
    edge_index = torch.tensor([], dtype=torch.long).view(2, 0)  # No edges in a single-node graph
    
    data = Data(x=features_tensor, edge_index=edge_index)
    
    # Predict the label and confidence
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        confidence = F.softmax(output, dim=1).max().item()  # Confidence is the highest probability
    
    # Return the predicted class (0 or 1) and the confidence
    return pred.item(), confidence

# Load the trained model
model = TargetModel(input_dim=feature_dim, hidden_dim=128)  # Adjust the input size
model.load_state_dict(torch.load('./data/saved_models/target_model.pth'))
print("Model loaded for prediction.")

# Take input from the user
user_sequence = input("Enter a protein sequence for prediction: ")

# Predict and output the result
prediction, confidence = predict_protein_sequence(user_sequence, model)
if prediction == 1:
    print(f"The protein sequence is predicted as positive (label 1) with a confidence of {confidence:.4f}.")
else:
    print(f"The protein sequence is predicted as negative (label 0) with a confidence of {confidence:.4f}.")
