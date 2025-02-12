# Early-drug-discovery-using-GCN

# Protein-Ligand Interaction Prediction

This project is a web-based tool for predicting ADMET properties and other pharmacological characteristics of protein-ligand interactions using a backend API.

## Features
- Accepts a protein sequence and a SMILES string as input.
- Sends data to a backend API for analysis.
- Displays ADMET predictions, hit identification, lead optimization, and target identification results.

## Installation
### Prerequisites
- Node.js
- Python (if backend setup is required)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```
2. Install dependencies (if applicable).

## Usage
1. Run the backend API (ensure it's running at `http://127.0.0.1:5000/analyze`).
2. Open `index.html` in a browser.
3. Enter a protein sequence and SMILES string.
4. Click "Predict" to analyze.
5. View results on the page.

## API Endpoint
- `POST http://127.0.0.1:5000/analyze`
  - Request Body:
    ```json
    {
      "protein_seq": "<your_protein_sequence>",
      "smiles": "<your_smiles_string>"
    }
    ```
  - Response:
    ```json
    {
      "report": {
        "ADMET Predictions": [...],
        "Hit Identification": "...",
        "Lead Optimization": {"best_molecule": "...", "best_qed_score": "..."},
        "Target Identification": "..."
      }
    }
    ```

## Contributing
Feel free to submit issues or pull requests to improve this tool.

## License
MIT License.

