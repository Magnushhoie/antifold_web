antifold_web
==============================

### Setup and running
```bash
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch -y  ## very important to specify pytorch package!
conda install pyg -c pyg -c conda-forge -y ## very important to make sure pytorch and cuda versions not being changed
conda install pip -y
pip install biotite pandas numpy

# Unzip model
unzip model.zip

# Run on example pdbs
python src/antifold.py \
    --pdb_csv data/example_pdbs.csv \
    --pdb_dir data/pdbs \
    --out_dir output/

# Example output
output/6y1l_imgt.csv

# Example notebook
notebooks/main.ipynb
```