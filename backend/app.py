from flask import Flask, request, jsonify
import torch
from torch.nn import functional as F
from backend.Targetmodel import TargetModel, HitModel, ADMETModel
from backend.preprocess import preprocess_molecule, preprocess_protein
from backend.utils import load_model, lead_optimization_process, optimize_user_input,generate_admet_output
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import random
import deepchem as dc
from flask_cors import CORS
from collections import Counter
from torch_geometric.data import Data
from rdkit.Chem import AllChem, DataStructs
from deepchem.feat import ConvMolFeaturizer
import pickle
import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)
CORS(app)

# Load pre-trained models
input_dim_target = 8010
hidden_dim_target = 128
input_dim_hit = 2048
hidden_dim_hit = 128
input_dim_admet = 1024
hidden_dim_admet = 128
kmer_vocab_file = '/Users/pavandevl/Desktop/Early_ddrug/data/kmer_vocab.txt'

# Assuming paths to pre-trained models are updated
try:
    target_model = load_model(TargetModel, "/Users/pavandevl/Desktop/Early_ddrug/data/saved_models/target_model.pth", input_dim=input_dim_target, hidden_dim=hidden_dim_target)
    hit_model = load_model(HitModel, "/Users/pavandevl/Desktop/Early_ddrug/data/saved_models/hit_model.pth", input_dim=input_dim_hit, hidden_dim=hidden_dim_hit)
    admet_model_dir = "/Users/pavandevl/Desktop/Early_ddrug/data/saved_models/ADMETpredict"
    admet_model = dc.deepchem.models.GraphConvModel(
        n_tasks=12,
        mode='classification',
        batch_normalize=False  # Match this to the saved model's configuration
    )
    # Restore the model from the saved checkpoint
    admet_model.restore(model_dir=admet_model_dir)
except Exception as e:
    print(f"Error loading models: {e}")

LEAD_data_path = '/Users/pavandevl/Desktop/Early_ddrug/data/250k_rndm_zinc_drugs_clean_3.csv'
def load_dataset_LEAD(file_path):
    df = pd.read_csv(file_path)
    smiles_list = df['smiles']  # Replace 'smiles' with the column name containing SMILES strings
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]
    
def load_tasks(filename="tasks.pkl"):
  with open(filename, "rb") as f:
    return pickle.load(f)

tasks = load_tasks('/Users/pavandevl/Desktop/Early_ddrug/data/tasks.pkl')

# Target identification Phase:
def kmer_count(sequence, k=3):
    """Extract k-mer counts from a sequence."""
    return Counter([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

def predict_protein_sequence(sequence, model, k=3):
    # Encode the input sequence
    kmer_counts = kmer_count(sequence, k)

    with open(kmer_vocab_file, 'r') as file:
        all_kmers = [line.strip() for line in file.readlines()]
    # all_kmers = sorted(kmer_counts.keys())  # Should be consistent with training k-mers
    kmer_to_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    feature_dim = len(all_kmers)
    
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

# HIT identification phase:
def predict_molecule(smiles, model):
    # Preprocess the input SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("Invalid SMILES string.")
        return None
    
    # Generate molecular fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = torch.zeros((1, 2048), dtype=torch.float)
    DataStructs.ConvertToNumpyArray(fp, arr.numpy())
    
    # Create a PyTorch Geometric Data object for the molecule
    x = arr
    num_nodes = x.shape[0]
    
    # Define edge_index as a dummy fully connected graph
    if num_nodes > 1:
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    
    # Create a data object
    data = Data(x=x, edge_index=edge_index)
    data = data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Predict using the model
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1).item()
        probability = F.softmax(output, dim=1)[0, pred].item()
    
    # Map the prediction back to the activity outcome
    activity = "Active HIT" if pred == 1 else "Inactive (Not a HIT)"
    return activity, probability

#LEAD Optimization phase:
def LEAD_optimize(LEAD_data_path,input_molecule):
    lead_data = load_dataset_LEAD(LEAD_data_path)
    best_molecule, best_score, history = optimize_user_input(lead_data,input_molecule)
    return best_molecule, best_score, history

#ADMET phase:
def predict_molecule_activity(smiles_input, model, tasks):
    """
    Predicts the activity of a molecule given a SMILES string using the trained model.
    
    Parameters:
    - smiles_input (str): The SMILES string of the molecule to be predicted.
    - model (GraphConvModel): The trained GraphConv model.
    - tasks (list): List of prediction tasks (e.g., ['Task1', 'Task2', ...]).
    
    Returns:
    - predictions (dict): A dictionary of task names and their corresponding predicted probabilities.
    """
    try:
        # Convert the SMILES string to an RDKit molecule
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        
        # Featurize the molecule using the ConvMolFeaturizer
        featurizer = ConvMolFeaturizer()
        mol_features = featurizer.featurize([mol])
        
        # Wrap the features in a NumpyDataset
        dataset = dc.data.NumpyDataset(X=mol_features)
        
        # Predict the properties for the input molecule using the model
        predictions = model.predict(dataset)
        
        # Ensure predictions have the correct shape
        if len(predictions.shape) != 3 or predictions.shape[0] != 1 or predictions.shape[2] != 2:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
        
        # Prepare the output in a dictionary format
        prediction_results = {}
        for i, task in enumerate(tasks):
            # Extract the probability of the positive class (class 1)
            prediction_results[task] = float(predictions[0, i, 1])  # Class 1 probability
        
        return prediction_results

    except Exception as e:
        return {"error": str(e)}

@app.route('/analyze', methods=['POST'])
def analyze_all_phases():
    try:
        data = request.get_json()
        protein_sequence = data.get('protein_seq')
        molecule_smiles = data.get('smiles')

        if not protein_sequence or not molecule_smiles:
            return jsonify({"error": "Protein sequence and molecule SMILES are required."}), 400

        # Target identification
        target_class, target_confidence = predict_protein_sequence(protein_sequence,target_model,k=3)
        print(target_class)
        # Hit identification
        hit_activity_class, probability = predict_molecule(molecule_smiles,hit_model)
        print(hit_activity_class)
        # Lead optimization (simplified for this integration)
        best_molecule, best_score, history = LEAD_optimize(LEAD_data_path,molecule_smiles)
        best_molecule_smiles = Chem.MolToSmiles(best_molecule)
        print(best_molecule_smiles)
        # ADMET predictions using the ADMET model
        ADMET_prediction_results = predict_molecule_activity(best_molecule_smiles, admet_model, tasks)

        detail_ADMET_output = generate_admet_output(ADMET_prediction_results)
        # Consolidate results
        report = {
            "Target Identification": f"It is a Active Target with confidence of {target_confidence*100}%" if target_class == 1 else "Inactive Target",
            "Hit Identification": str(hit_activity_class)+ f"with confidence of {probability*100}%" if hit_activity_class=='Active HIT' else str(hit_activity_class),
            "Lead Optimization": {"best_molecule":best_molecule_smiles,"best_qed_score":best_score,"history":history},
            "ADMET Predictions": detail_ADMET_output
        }
        print(report)

        return jsonify({"report": report}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
