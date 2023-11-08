AntiFold
==============================

Code for AntiFold paper, accepted for [NeurIPS 2023 GenBio spotlight](https://openreview.net/forum?id=bxZMKHtlL6)

Webserver: [OPIG webserver](https://opig.stats.ox.ac.uk/webapps/AntiFold/)
Code: [antifold_code.zip](https://opig.stats.ox.ac.uk/data/downloads/AntiFold/antifold_code.zip)
Model: [model.pt](https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt)

# Colab
To test the method out without installing it you can try this: [![Open In Colab](images/colab-badge.svg)](https://colab.research.google.com/drive/1TTfgjoZx3mzF5u4e9b4Un9Y7b_rqXc_4)

## Quickstart guide

See <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/notebook.ipynb">Jupyter notebook</a> or follow quickstart guide with example PDBs:

```bash
# Download AntiFold model and code (Linux)
mkdir -p antifold/models && cd antifold
wget https://opig.stats.ox.ac.uk/data/downloads/AntiFold/antifold.zip
wget -P models/ https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt
unzip antifold.zip

# Setup environment and install AntiFold (GPU)
# Nb: For CPU use: conda install -c pytorch pytorch
conda create --name antifold python=3.9 -y
conda activate antifold
conda install -c conda-forge pytorch-gpu # cudatoolkit=11.3 recommended
conda install -c pyg pyg -y
conda install -c conda-forge pip -y

# Install AntiFold
pip install .
```

```bash
# Run on single PDB, CDRH3 only
python antifold/main.py \
    --out_dir output/single_pdb \
    --pdb_file data/pdbs/6y1l_imgt.pdb \
    --heavy_chain H \
    --light_chain L \
    --num_seq_per_target 10 \
    --sampling_temp "0.2" \
    --regions "CDRH3"

# Run on example pdbs, all CDRs, temperatures 0.20 and 0.30
python antifold/main.py \
    --out_dir output/example_pdbs \
    --pdbs_csv data/example_pdbs.csv \
    --pdb_dir data/pdbs \
    --num_seq_per_target 10 \
    --sampling_temp "0.20 0.30" \
    --regions "CDR1 CDR2 CDR3"

# Extract (ESM-IF1) embeddings with custom chains
python antifold/main.py \
    --out_dir output/untested/ \
    --pdbs_csv data/untested.csv \
    --pdb_dir data/untested/ \
    --model_path "ESM-IF1" \
    --custom_chain_mode \
    --extract_embeddings
```

## Example output
Output CSV with residue log-probabilities: Residue probabilities: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/output/example_pdbs/6y1l_imgt.csv">6y1l_imgt.csv</a>
```csv
pdb_pos,pdb_chain,aa_orig,aa_pred,pdb_posins,perplexity,A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
2,H,V,M,2,1.6488,-4.9963,-6.6117,-6.3181,-6.3243,-6.7570,-4.2518,-6.7514,-5.2540,-6.8067,-5.8619,-0.0904,-6.5493,-4.8639,-6.6316,-6.3084,-5.1900,-5.0988,-3.7295,-8.0480,-7.3236
3,H,Q,Q,3,1.3889,-10.5258,-12.8463,-8.4800,-4.7630,-12.9094,-11.0924,-5.6136,-10.9870,-3.1119,-8.1113,-9.4382,-6.2246,-13.3660,-0.0701,-4.9957,-10.0301,-6.8618,-7.5810,-13.6721,-11.4157
4,H,L,L,4,1.0021,-13.3581,-12.6206,-17.5484,-12.4801,-9.8792,-13.6382,-14.8609,-13.9344,-16.4080,-0.0002,-9.2727,-16.6532,-14.0476,-12.5943,-15.4559,-16.9103,-17.0809,-10.5670,-13.5334,-13.4324
...
```

Output FASTA file with sampled sequences: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/output/example_pdbs/6y1l_imgt.fasta">6y1l_imgt.fasta</a>
- Score: average log-odds of residues in the sampled region
- Global: average log-odds of all residues (IMGT positions 1-128)
```fasta
>6y1l_imgt , score=0.2934, global_score=0.2934, regions=['CDR1', 'CDR2', 'CDRH3'], model_name=AntiFold, seed=42
VQLQESGPGLVKPSETLSLTCAVSGYSISSGYYWGWIRQPPGKGLEWIGSIYHSGSTYYN
PSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAGLTQSSHNDANWGQGTLVTVSS/V
LTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKRLIYDNNKRPSGIPDRF
SGSKSGTSATLGITGLQTGDEADYYCGTWDSSLNPVFGGGTKLEIKR
> T=0.20, sample=1, score=0.3930, global_score=0.1869, seq_recovery=0.8983, mutations=12
VQLQESGPGLVKPSETLSLTCAVSGASITSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYN
PSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCAGLYGSPWSNPYWGQGTLVTVSS/V
LTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKRLIYDNNKRPSGIPDRF
SGSKSGTSATLGITGLQTGDEADYYCGTWDSSLNPVFGGGTKLEIKR
...
```
## Example notebook
Notebook: <a href="https://opig.stats.ox.ac.uk/data/downloads/AntiFold/notebook.ipynb">notebook.ipynb</a>

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

## Usage
```bash
usage: 
    # Predict on example PDBs in folder
    python antifold/main.py     --pdbs_csv data/example_pdbs.csv     --pdb_dir data/pdbs     --out_dir output/
    
Predict antibody variable domain, inverse folding probabilities and sample sequences with maintained fold.
PDB structures should be IMGT-numbered, paired heavy and light chain variable domains (positions 1-128).

For IMGT numbering PDBs use SAbDab or https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarci/

options:
  -h, --help            show this help message and exit
  --pdb_file PDB_FILE   Input PDB file (for single PDB predictions)
  --heavy_chain HEAVY_CHAIN
                        Ab heavy chain (for single PDB predictions)
  --light_chain LIGHT_CHAIN
                        Ab light chain (for single PDB predictions)
  --antigen_chain ANTIGEN_CHAIN
                        Antigen chain (experimental)
  --pdbs_csv PDBS_CSV   Input CSV file with PDB names and H/L chains (multi-PDB predictions)
  --pdb_dir PDB_DIR     Directory with input PDB files (multi-PDB predictions)
  --out_dir OUT_DIR     Output directory
  --regions REGIONS     Space-separated regions to mutate. Default 'CDR1 CDR2 CDR3H'
  --num_seq_per_target NUM_SEQ_PER_TARGET
                        Number of sequences to sample from each antibody PDB
  --sampling_temp SAMPLING_TEMP
                        A string of temperatures e.g. '0.20 0.25 0.50' (default 0.20). Sampling temperature for amino acids. Suggested values 0.10, 0.15, 0.20, 0.25, 0.30. Higher values will lead to more diversity.
  --limit_variation     Limit variation to as many mutations as expected from temperature sampling
  --extract_embeddings  Extract per-residue embeddings from AntiFold / ESM-IF1
  --custom_chain_mode   Custom chain input (experimental, e.g. single chain, inclusion of antigen chain or any chains with ESM-IF1)
  --exclude_heavy       Exclude heavy chain from sampling
  --exclude_light       Exclude light chain from sampling
  --batch_size BATCH_SIZE
                        Batch-size to use
  --num_threads NUM_THREADS
                        Number of CPU threads to use for parallel processing (0 = all available)
  --seed SEED           Seed for reproducibility
  --model_path MODEL_PATH
                        AntiFold model weights. Set to 'ESM-IF1' to use ESM-IF1 model instead of AntiFold fine-tuned model
  --verbose VERBOSE     Verbose printing
```

## IMGT regions dict
```python
IMGT_dict = {
    "all": range(1, 128 + 1),
    "allH": range(1, 128 + 1),
    "allL": range(1, 128 + 1),
    "FWH": list(range(1, 26 + 1)) + list(range(40, 55 + 1)) + list(range(66, 104 + 1)),
    "FWL": list(range(1, 26 + 1)) + list(range(40, 55 + 1)) + list(range(66, 104 + 1)),
    "CDRH": list(range(27, 39)) + list(range(56, 65 + 1)) + list(range(105, 117 + 1)),
    "CDRL": list(range(27, 39)) + list(range(56, 65 + 1)) + list(range(105, 117 + 1)),
    "FW1": range(1, 26 + 1),
    "FWH1": range(1, 26 + 1),
    "FWL1": range(1, 26 + 1),
    "CDR1": range(27, 39),
    "CDRH1": range(27, 39),
    "CDRL1": range(27, 39),
    "FW2": range(40, 55 + 1),
    "FWH2": range(40, 55 + 1),
    "FWL2": range(40, 55 + 1),
    "CDR2": range(56, 65 + 1),
    "CDRH2": range(56, 65 + 1),
    "CDRL2": range(56, 65 + 1),
    "FW3": range(66, 104 + 1),
    "FWH3": range(66, 104 + 1),
    "FWL3": range(66, 104 + 1),
    "CDR3": range(105, 117 + 1),
    "CDRH3": range(105, 117 + 1),
    "CDRL3": range(105, 117 + 1),
    "FW4": range(118, 128 + 1),
    "FWH4": range(118, 128 + 1),
    "FWL4": range(118, 128 + 1),
}
```
