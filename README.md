antifold_web
==============================

Code for AntiFold paper, submitted for NeurIPS 2023.

Code and model: [antifold.zip](https://opig.stats.ox.ac.uk/data/downloads/AntiFold/antifold.zip)

## Quickstart guide

See <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/colab.ipynb">Google Colab</a> or follow quickstart guide with example PDBs:

```bash
# Download
mkdir antifold && cd antifold
wget https://opig.stats.ox.ac.uk/data/downloads/AntiFold/antifold.zip
unzip antifold.zip

# Setup environment and install AntiFold
conda create --name antifold python=3.9 -y
conda activate antifold
conda install pytorch -c pytorch -y # cudatoolkit=11.3 recommended
conda install -c pyg pyg -y
conda install -c conda-forge pip -y

# Install AntiFold
pip install .

# Run on example pdbs
python antifold/main.py \
    --pdbs_csv data/example_pdbs.csv \
    --sampling_temp 0.2 \
    --regions CDR1 CDR2 CDR3H \
    --pdb_dir data/pdbs \
    --out_dir output/example_pdbs
```

## Usage
```python
import pandas as pd

# Put IMGT numbered PDBs (Fv only, IMGT position 1-128) to process and load a CSV file with PDB names and heavy/light chains
# Define the PDB and chains in DataFrame
pdb_dir = "data/pdbs"
df_pdbs = pd.read_csv("data/example_pdbs.csv")

# Regions to mutate (IMGT)
regions_to_mutate = ["CDR1", "CDR2", "CDR3H"]

# Load model
import antifold.main as antifold
model = antifold.load_IF1_model("models/model.pt")

# Sample from PDBs, 10 sequences each at temperature 0.50 in regions CDR1, CDR2, CDR3H
pdb_output_dict = antifold.sample_pdbs(
                    model,
                    pdbs_csv_or_dataframe=df_pdbs, # Path to CSV file, or a DataFrame
                    regions_to_mutate=regions_to_mutate,
                    pdb_dir="data/pdbs",
                    sample_n=10,
                    sampling_temp=0.50,
                    limit_expected_variation=False
                    )

# Output dictionary with sequences, and residue probabilities or log-odds
pdbs = pdb_output_dict.keys()

# Residue log probabilities
df_logprobs = pdb_output_dict["6y1l_imgt"]["logprobs"]

# Sampled sequences
fasta_dict = pdb_output_dict["6y1l_imgt"]["sequences"]
```

## Example output
Output CSV with residue log-probabilities: Residue probabilities: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/output/example_pdbs/6y1l_imgt.csv">6y1l_imgt.csv</a>
```csv
pdb_pos,pdb_chain,aa_orig,aa_pred,pdb_posins,perplexity,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
2,H,V,M,2,1.7221,-4.9226,-6.4708,-6.1467,-6.1754,-6.5648,-4.2163,-6.5528,-5.1208,-6.6035,-5.7443,-0.0992,-6.2694,-4.8115,-6.4361,-6.1866,-4.9160,-4.9723,-3.7213,-7.9959,-7.1096
3,H,Q,Q,3,1.8164,-9.5358,-12.3109,-8.1272,-4.2464,-12.4795,-10.1501,-5.2837,-10.4261,-2.0793,-7.4281,-8.6754,-5.7362,-12.2200,-0.1787,-4.4097,-9.2673,-6.3677,-7.0966,-13.2125,-11.0414
4,H,L,L,4,1.0017,-13.7056,-12.7263,-17.5119,-12.3137,-10.0561,-13.7474,-14.7022,-14.1318,-16.2906,-0.0001,-9.5772,-16.8369,-14.1303,-12.3832,-15.3246,-16.9911,-17.1241,-10.8850,-13.3156,-13.4724
```

Output FASTA file with sampled sequences: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/output/example_pdbs/6y1l_imgt.fasta">6y1l_imgt.fasta</a>
- Score: average log-odds of residues in the sampled region
- Global: average log-odds of all residues (IMGT positions 1-128)
```fasta
>6y1l_imgt , score=0.3504, global_score=0.3504, regions=['CDR1', 'CDR2', 'CDR3H'], model_name=AntiFold, seed=42
VQLQESGPGLVKPSETLSLTCAVSGYSISSGYYWGWIRQPPGKGLEWIGSIYHSGSTYYN
PSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAGLTQSSHNDANWGQGTLVTVSS/V
LTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKRLIYDNNKRPSGIPDRF
SGSKSGTSATLGITGLQTGDEADYYCGTWDSSLNPVFGGGTKLEIKR
> T=0.50, sample=1, score=0.5675, global_score=0.2636, seq_recovery=0.8898, mutations=13
VQLQESGPGLVKPSETLSLTCAVSGASITSSYYWGWIRQPPGKGLEWIGSIYYSGTTYYN
PSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAGLYGSPYSTPAWGQGTLVTVSS/V
LTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKRLIYDNNKRPSGIPDRF
SGSKSGTSATLGITGLQTGDEADYYCGTWDSSLNPVFGGGTKLEIKR
```

## Example notebook
Notebook: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/colab.ipynb">colab.ipynb</a>

## Usage
```bash
usage: 
    # Predict on example PDBs in folder
    python antifold/main.py     --pdbs_csv data/example_pdbs.csv     --pdb_dir data/pdbs     --out_dir output/
    

Predict antibody variable domain, inverse folding probabilities and sample sequences with maintained fold.
Requires IMGT-numbered PDBs with paired heavy and light chains.

optional arguments:
  -h, --help            show this help message and exit
  --pdbs_csv PDBS_CSV   Input CSV file with PDB names and H/L chains
  --pdb_dir PDB_DIR     Directory with input PDB files
  --out_dir OUT_DIR     Output directory
  --regions REGIONS [REGIONS ...]
                        Space-separated list of regions to mutate (e.g., CDR1 CDR2 CDR3H).
  --num_seq_per_target NUM_SEQ_PER_TARGET
                        Number of sequences to sample from each antibody PDB
  --sampling_temp SAMPLING_TEMP [SAMPLING_TEMP ...]
                        A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.
  --limit_variation     Limit variation to as many mutations as expected from temperature sampling
  --exclude_heavy       Exclude heavy chain from sampling
  --exclude_light       Exclude light chain from sampling
  --batch_size BATCH_SIZE
                        Batch-size to use
  --seed SEED           Seed for reproducibility
  --model_path MODEL_PATH
                        Output directory
  --verbose VERBOSE     Verbose printing
```