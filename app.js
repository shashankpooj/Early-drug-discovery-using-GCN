document.getElementById('predict').addEventListener('click', async () => {
    const protein_seq = document.getElementById('protein_seq').value.trim();
    const smiles = document.getElementById('smiles').value.trim();

    if (!protein_seq || !smiles) {
        document.getElementById('result').innerHTML = '<p style="color: red;">Please provide both Protein Sequence and SMILES string.</p>';
        return;
    }

    document.getElementById('result').innerHTML = '<div class="loading">Loading...</div>';

    try {
        const response = await fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ protein_seq, smiles })
        });

        const data = await response.json();
        console.log('Backend Report:', data.report);

        if (response.ok) {
            const report = data.report;

            if (report && report['ADMET Predictions']) {
                let admetPredictions = report['ADMET Predictions'];
                let reportContent = `
                    <h3>Prediction Report</h3>
                    <h4>ADMET Predictions</h4>
                    <ul>
                        <li><strong>Nuclear Receptor - Androgen Receptor:</strong> ${admetPredictions[0]}</li>
                        <li><strong>Nuclear Receptor - Androgen Receptor Ligand Binding Domain:</strong> ${admetPredictions[1]}</li>
                        <li><strong>Nuclear Receptor - Aryl Hydrocarbon Receptor:</strong> ${admetPredictions[2]}</li>
                        <li><strong>Nuclear Receptor - Aromatase (involved in estrogen synthesis):</strong> ${admetPredictions[3]}</li>
                        <li><strong>Nuclear Receptor - Estrogen Receptor:</strong> ${admetPredictions[4]}</li>
                        <li><strong>Nuclear Receptor - Estrogen Receptor Ligand Binding Domain:</strong> ${admetPredictions[5]}</li>
                        <li><strong>Nuclear Receptor - Peroxisome Proliferator-Activated Receptor Gamma:</strong> ${admetPredictions[6]}</li>
                        <li><strong>Steroid Receptor - Antioxidant Response Element:</strong> ${admetPredictions[7]}</li>
                        <li><strong>Steroid Receptor - ATPase Family AAA Domain Containing 5:</strong> ${admetPredictions[8]}</li>
                        <li><strong>Steroid Receptor - Heat Shock Element:</strong> ${admetPredictions[9]}</li>
                        <li><strong>Steroid Receptor - Matrix Metalloproteinase:</strong> ${admetPredictions[10]}</li>
                        <li><strong>Steroid Receptor - p53 (Tumor Suppressor Protein):</strong> ${admetPredictions[11]}</li>
                    </ul>
                    <h4>Hit Identification</h4>
                    <p><strong>Status:</strong> ${report['Hit Identification'] || 'Not Available'}</p>
                    <h4>Lead Optimization</h4>
                    <p><strong>Best Molecule:</strong> ${report['Lead Optimization']?.best_molecule || 'Not Available'}</p>
                    <p><strong>Best QED Score:</strong> ${report['Lead Optimization']?.best_qed_score || 'Not Available'}</p>
                    <h4>Target Identification</h4>
                    <p><strong>Status:</strong> ${report['Target Identification'] || 'Not Available'}</p>
                `;
                document.getElementById('result').innerHTML = reportContent;
            } else {
                document.getElementById('result').innerHTML = '<p style="color: red;">Error: Missing or invalid ADMET Predictions data from backend.</p>';
            }
        } else {
            document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${data.error || 'Unknown error from backend.'}</p>`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = `<p style="color: red;">Error: Unable to process the request. ${error.message}</p>`;
    }
});
