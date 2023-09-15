antifold_web
==============================

# Quickstart guide

```bash
# Setup environment and install
conda create --name inverse python=3.9 -y
conda activate inverse
conda install -c pyg pyg -y
conda install -c conda-forge pip -y

git clone https://github.com/Magnushhoie/discotope3_web/
cd discotope3_web/
pip install .

# Unzip models to use (TODO)
unzip models.zip

# Run on example pdbs
python antifold/main.py \
    --pdb_csv data/example_pdbs.csv \
    --pdb_dir data/pdbs \
    --out_dir output/

# Example output
output/6y1l_imgt.csv

# Example notebook
notebooks/main.ipynb
```