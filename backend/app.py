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
kmer_vocab_file = '/Users/shashank/Desktop/Early_ddrug/data/kmer_vocab.txt'

# Assuming paths to pre-trained models are updated

target_model = load_model(TargetModel, "/Users/shashank/Desktop/Early_ddrug/data/saved_models/target_model.pth", input_dim=input_dim_target, hidden_dim=hidden_dim_target)
hit_model = load_model(HitModel, "/Users/shashank/Desktop/Early_ddrug/data/saved_models/hit_model.pth", input_dim=input_dim_hit, hidden_dim=hidden_dim_hit)
    

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
    edge_index = torch.tensor(features_tensor , dtype=torch.long).view(2, 0)  # No edges in a single-node graph
    
    data = Data(x=features_tensor, edge_index=edge_index)
    
    # Predict the label and confidence
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1).item()
        confidence = F.softmax(output) # Confidence is the highest probability
         
    # Return the predicted class (0 or 1) and the confidence
    return pred,confidence


# HIT identification phase:
def predict_molecule(smiles, model):
    # Preprocess the input SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("Invalid SMILES string.")
        return None
    
    # Generate molecular fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
   
    arr=DataStructs.ConvertToNumpyArray(fp.numpy())
    
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
 
    
    # Predict using the model
    model.eval()
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1).item()
        probability = F.softmax(output, dim=1)[0, pred].item()
    
    # Map the prediction back to the activity outcome
    activity = "Active HIT" if pred == 1 else "Inactive (Not a HIT)"
    return activity, probability

import random

def lead_optimize(input_mol):
    # Possible modifications to the molecule
    modifications = ['-OH', '-COOH', '-N2']

    # Initialize best molecule and its score
    best_mol = input_mol
    input_mol_score = QED_score(input_mol)  # Replace with actual scoring function

    # Try modifications 3 times
    for _ in range(3):
        new_mol = best_mol + random.choice(modifications)
        # Compute new score (QED or any scoring metric)
        new_score = QED_score(new_mol)  # Replace with actual function

        # Update best molecule if new score is better
        if new_score > input_mol_score:
            best_mol = new_mol
            input_mol_score = new_score

    return best_mol



#ADMET phase:
# 
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
