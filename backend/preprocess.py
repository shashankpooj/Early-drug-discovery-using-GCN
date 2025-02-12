from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import torch
from torch_geometric.data import Data
from collections import Counter


def preprocess_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    adj = rdmolops.GetAdjacencyMatrix(mol)
    features = torch.eye(adj.shape[0])  # One-hot encoding of atoms
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

    return Data(x=features, edge_index=edge_index)

def preprocess_protein(sequence,k,known_kmers=None):
   """Encode a protein sequence into a feature vector using k-mer counts.
    
    Args:
        sequence (str): Protein sequence to encode.
        k (int): Length of k-mers.
        known_kmers (set, optional): Set of known k-mers. If None, compute dynamically.

    Returns:
        torch.Tensor: Encoded feature vector.
        dict: Mapping of k-mers to their respective indices.
    """
   def kmer_count(sequence, k):
        return Counter([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

    # Compute k-mer counts for the sequence
   kmer_counts = kmer_count(sequence, k)

    # Generate k-mer index if not provided
   if known_kmers is None:
        all_kmers = sorted(kmer_counts.keys())
        kmer_to_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
   else:
        kmer_to_index = {kmer: idx for idx, kmer in enumerate(sorted(known_kmers))}

    # Initialize feature vector
   feature_dim = len(kmer_to_index)
   features = [0] * feature_dim

   for kmer, count in kmer_counts.items():
        if kmer in kmer_to_index:
            features[kmer_to_index[kmer]] = count

   return torch.tensor(features, dtype=torch.float), kmer_to_index
