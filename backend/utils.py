import torch 
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import random


def load_model(model_class, model_path, *args, **kwargs):
    """Load a trained model from disk."""
    model = model_class(*args, **kwargs)  # Pass arguments to the model's constructor
    model.load_state_dict(torch.load(model_path,weights_only=True),strict=False)
    model.eval()
    return model

# Reward function based on QED score
def qed_reward(molecule):
    try:
        return QED.qed(molecule)
    except Exception:
        return 0.0  # Handle cases where QED calculation fails

# Mutation function: Randomly modify molecules
def mutate_molecule(molecule):
    try:
        editable_mol = Chem.RWMol(molecule)
        mutation_type = random.choice(['add_atom', 'remove_atom', 'add_bond', 'remove_bond'])

        if mutation_type == 'remove_atom':
            atom_idx = random.choice(range(molecule.GetNumAtoms()))
            editable_mol.RemoveAtom(atom_idx)
        elif mutation_type == 'add_atom':
            new_atom = Chem.Atom(random.choice(['C', 'N', 'O', 'F']))
            editable_mol.AddAtom(new_atom)
        elif mutation_type == 'add_bond':
            atom_indices = random.sample(range(editable_mol.GetNumAtoms()), 2)
            editable_mol.AddBond(atom_indices[0], atom_indices[1], Chem.rdchem.BondType.SINGLE)
        elif mutation_type == 'remove_bond':
            bonds = list(editable_mol.GetBonds())
            bond_to_remove = random.choice(bonds)
            editable_mol.RemoveBond(bond_to_remove.GetBeginAtomIdx(), bond_to_remove.GetEndAtomIdx())

        return editable_mol.GetMol()
    except Exception:
        return molecule  # Return original molecule if mutation fails

# Lead optimization loop
def lead_optimization_process(dataset, iterations=100):
    best_molecule = random.choice(dataset)  # Start with a random molecule
    best_score = qed_reward(best_molecule)
    history = []  # Track optimization history

    for iteration in range(iterations):
        mutated_molecule = mutate_molecule(best_molecule)
        mutated_score = qed_reward(mutated_molecule)

        # Update if the mutated molecule is better
        if mutated_score > best_score:
            best_molecule = mutated_molecule
            best_score = mutated_score

        # Log history
        history.append((Chem.MolToSmiles(best_molecule), best_score))

    return best_molecule, best_score, history

def optimize_user_input(dataset,user_input,iterations=100):
    # Convert the input to a molecule object
    input_molecule = Chem.MolFromSmiles(user_input)

    if input_molecule is None:
        print("Invalid SMILES string!")
        return None, None, None  # Return None if the SMILES string is invalid
    
    input_score = qed_reward(input_molecule)
    print(f"Input molecule: {user_input} has QED score: {input_score:.4f}")

    # Perform lead optimization on the user input molecule
    best_molecule, best_score, history = lead_optimization_process([input_molecule] + dataset, iterations)

    # Print the optimized molecule and its QED score
    print(f"Best molecule: {Chem.MolToSmiles(best_molecule)} with QED score: {best_score:.4f}")

    return best_molecule, best_score, history



# Mapping the receptors to ADMET properties
admet_properties = {
    'NR-AR': 'Absorption/Distribution',
    'NR-AR-LBD': 'Metabolism/Binding',
    'NR-AhR': 'Metabolism/Toxicity',
    'NR-Aromatase': 'Metabolism',
    'NR-ER': 'Absorption/Distribution',
    'NR-ER-LBD': 'Metabolism/Binding',
    'NR-PPAR-gamma': 'Distribution',
    'SR-ARE': 'Toxicity',
    'SR-ATAD5': 'Excretion/Toxicity',
    'SR-HSE': 'Toxicity/Stress Response',
    'SR-MMP': 'Excretion/Toxicity',
    'SR-p53': 'Toxicity'
}
# Function to classify the impact based on the score
def classify_impact(score):
    if score > 0.2:
        return 'Strong Impact'
    elif score > 0.1:
        return 'Moderate Impact'
    elif score > 0.05:
        return 'Low Impact'
    else:
        return 'Minimal Impact'

# Generate structured output
def generate_admet_output(task_scores):
    output = []
    for task, score in task_scores.items():
        property_influence = admet_properties.get(task)
        impact = classify_impact(score)
        output.append(f"{task}: {score:.4f} - {property_influence} ({impact})")
    return output