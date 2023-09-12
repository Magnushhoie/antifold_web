import glob
import os
from argparse import ArgumentParser, RawTextHelpFormatter

import numpy as np
import pandas as pd
import torch
from biotite.sequence.seqtypes import ProteinSequence
from biotite.structure.io import pdb
from biotite.structure.residues import get_residue_starts


def cmdline_args():
    # Make parser object
    usage = f"""
    # Convert ProteinMPNN npz files to csv files
    python mpnn_npzs_to_csvs.py \
    --pdb_dir pdbs/ \
    --npz_dir npzs/
    """
    p = ArgumentParser(
        description="",
        formatter_class=RawTextHelpFormatter,
        usage=usage,
    )

    p.add_argument(
        "--pdb_dir",
        required=True,
        help="PDB directory",
    )

    p.add_argument(
        "--npz_dir",
        required=True,
        help="NPZ file directory (output ProteinMPNN)",
    )

    p.add_argument("-v", "--verbose", type=int, default=0, help="Verbose logging")

    return p.parse_args()


def probs_to_df(probs_file):
    """Converts ProteinMPNN npy probs file to DataFrame"""
    npy_dict = np.load(probs_file)

    # Sequence and probs from log_probs
    seq_num_true = npy_dict["S"]
    # t_probs = torch.exp(torch.Tensor(npy_dict["log_p"][0]))
    t_probs = torch.Tensor(npy_dict["log_p"][0])

    # DataFrame
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    df_probs = pd.DataFrame(data=t_probs, columns=list(alphabet), index=seq_num_true)

    return df_probs


def df_to_seq_recovery_rat(df_probs):
    """Calculates recovered sequence ratio from probs df"""
    seq_recovery_rat = (df_probs.values.argmax(axis=1) == df_probs.index).sum() / len(
        df_probs
    )
    return seq_recovery_rat


def biotite_read_pdb_res_posins_chain(pdb_file):
    """Read a PDB file and return the IMGT positions"""

    def get_res_posins(array):
        """Get the residues and position+insertion codes (IMGT) of a Biotite atom array (stack)"""
        starts = get_residue_starts(array)

        # Get 1-letter amino acids (int)
        res_3letter = array.res_name[starts]
        res_aa = np.array(
            [ProteinSequence.convert_letter_3to1(aa) for aa in res_3letter]
        )

        # Get IMGT insertion code and position (str)
        res_pos = array.res_id[starts]
        res_ins = array.ins_code[starts]
        res_chain = array.chain_id[starts]
        res_posins = np.array(
            list([f"{pos}{ins}" for pos, ins in zip(res_pos, res_ins)])
        )

        return res_aa, res_posins, res_chain

    # Read PDB, get structure
    pdbf = pdb.PDBFile.read(pdb_file)
    structure = pdb.get_structure(pdbf, model=1)

    # Get IMGT positions and amino acids
    res_aa, res_posins, res_chain = get_res_posins(structure)

    return res_aa, res_posins, res_chain


def mpnn_npz_pdb_to_df(npz_file, pdb_file):
    """Converts ProteinMPNN npy probs file to DataFrame"""

    def get_reordered_mpnn_chains(pdb_chains):
        # MPNN sometimes reverses chain order by alphanumerical sorting (ED -> DE)
        # If sorted order != PDB order, then reverse back to PDB order
        pdb_order = pd.Series(pdb_chains).unique()
        mpnn_order = np.sort(pdb_order)
        assert len(pdb_order) == 2

        if (pdb_order != mpnn_order).all():
            # print("Reversing")
            mpnn_chains = np.sort(pdb_chains)
            H_idxs = np.where(mpnn_chains == mpnn_order[1])[0]
            L_idxs = np.where(mpnn_chains == mpnn_order[0])[0]
            mpnn_corrected = np.concatenate([H_idxs, L_idxs])
        else:
            mpnn_corrected = np.arange(len(pdb_chains))

        return mpnn_corrected

    def get_reordered_mpnn_112_indices(pdb_posins):
        """Splits 2-chain PDB positions into H + L chains, then reverses the 112 positions"""

        # Split chains, re-order position 112 and merge
        positions = pd.Series(pdb_posins).str.extract(r"(\d+)")[0].astype(int).values
        split_idx = int((np.where(np.diff(positions) < -50)[0] + 1)[0])

        # Gets indices up to 112, reverses 112, then continues normally
        H_pos = pdb_posins[:split_idx]
        H_112 = np.where(pd.Series(H_pos).str.startswith("112"))[0]
        if len(H_112) == 0:
            H_pos_new = np.array(range(len(H_pos)))
        else:
            H_pos_new = np.array(
                list(range(H_112.min()))  # Up to 112 positions
                + list(H_112[::-1])  # Reverse 112 positions
                + list(range(H_112.max() + 1, len(H_pos)))  # Continue normally after
            )
        # Now for L chain
        L_pos = pdb_posins[split_idx:]
        L_112 = np.where(pd.Series(L_pos).str.startswith("112"))[0]
        if len(L_112) == 0:
            L_pos_new = np.array(range(len(L_pos)))
        else:
            L_pos_new = np.array(
                list(range(L_112.min()))  # Up to 112 positions
                + list(L_112[::-1])  # Reverse 112 positions
                + list(range(L_112.max() + 1, len(L_pos)))  # Continue normally after
            )
        # Merge
        pdb_posins_new = np.concatenate([H_pos_new, L_pos_new + len(H_pos)])

        return pdb_posins_new

    # Read ProtienMPNN npz file (for probs)
    npy_dict = np.load(npz_file)

    # Sequence and probs from log_probs
    seq_num_true = npy_dict["S"]
    t_probs = torch.Tensor(npy_dict["log_p"][0])
    # Fix MPNN: Filter out "-" gap positions (ProteinMPNN adds gaps when jumps in position indices)
    t_probs = t_probs[seq_num_true != 20]
    seq_num_true = seq_num_true[seq_num_true != 20]

    # Read PDB file to align with it on chains and positions
    pdb_res, pdb_posins, pdb_chains = biotite_read_pdb_res_posins_chain(pdb_file)
    # Fix MPNN: MPNN sometimes reverses H+L order by alphanumerically sorting chains
    idxs = get_reordered_mpnn_chains(pdb_chains)
    t_probs = t_probs[idxs]
    seq_num_true = seq_num_true[idxs]
    # Fix MPNN: MPNN reverses 112 order by alphanumerically sorting position+insertion codes
    # IMGT correct (112B, 112A, 112) is incorrectly sorted as (112, 112A, 112B)
    idxs = get_reordered_mpnn_112_indices(pdb_posins)
    t_probs = t_probs[idxs]
    seq_num_true = seq_num_true[idxs]

    # Convert to DataFrame and columns
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"  # Nb: "X" should not be present anymore
    df_probs = pd.DataFrame(data=t_probs, columns=list(alphabet), index=seq_num_true)
    df_probs.insert(0, "aa_orig", [list(alphabet)[i] for i in seq_num_true])
    df_probs.insert(
        0,
        "aa_pred",
        [list(alphabet)[i] for i in df_probs[list(alphabet)].values.argmax(axis=1)],
    )
    df_probs.insert(2, "pdb_res", pdb_res)
    df_probs.insert(3, "pdb_posins", pdb_posins)
    df_probs.insert(4, "pdb_chain", pdb_chains)

    # Check that ProteinMPNN and Biotite sequence matches
    if (df_probs["aa_orig"].values != pdb_res).any():
        if (df_probs["aa_orig"].values != pdb_res).any():
            print(f"ProteinMPNN and Biotite IMGT positions do not match for {pdb_file}")
            # return df_probs
            raise Exception

    return df_probs


def mpnn_npzdir_to_csv(npz_dir, pdb_dir):
    """Recursively finds MPNN npz files, matches w/ PDB and saves probs CSV"""

    npz_files = glob.glob(f"{npz_dir}/**/*.npz", recursive=True)

    print(f"Found {len(npz_files)} npz files in {npz_dir} ...")
    print(f"Converting to CSV using PDB files from {pdb_dir} ...")

    for i, npz_file in enumerate(npz_files):
        # Try to find matching PDB file
        pdb_name = os.path.splitext(os.path.basename(npz_file))[0]
        pdb_file = f"{pdb_dir}/{pdb_name}.pdb"
        # Save to where .npz file is as .csv
        csv_file = os.path.join(
            os.path.dirname(npz_file),
            os.path.splitext(os.path.basename(npz_file))[0] + ".csv",
        )

        if not os.path.isfile(pdb_file):
            print(f"PDB {pdb_name}.pdb not found in pdb_dir for npz_file {npz_file}")
            raise Exception

        print(
            f"{i+1} / {len(npz_files)}: {pdb_name}.npz + {pdb_name}.pdb -> {csv_file}"
        )
        # Prepare DataFrame from npz (probs) and pdb (IMGT) files
        df_probs = mpnn_npz_pdb_to_df(npz_file, pdb_file)
        df_probs.to_csv(csv_file)


def main(args):
    _ = mpnn_npzdir_to_csv(args.npz_dir, args.pdb_dir)

    """
    # ImmuneBuilder structures
    pdb_dir = "/home/maghoi/repos/__personal/antifold/data/abmpnn/IB"
    # ProteinMPNN
    npz_dir = "/home/maghoi/repos/__personal/antifold/output/proteinmpnn/IB/"
    _ = mpnn_npzdir_to_csv(npz_dir, pdb_dir)
    # AbMPNN
    npz_dir = "/home/maghoi/repos/__personal/antifold/output/abmpnn/IB/"
    _ = mpnn_npzdir_to_csv(npz_dir, pdb_dir)

    # Solved structures
    # pdb_dir = "/home/maghoi/repos/__personal/antifold/data/abmpnn/SAB"
    # ProteinMPNN
    # npz_dir = "/home/maghoi/repos/__personal/antifold/output/proteinmpnn/SAB/"
    # _ = mpnn_npzdir_to_csv(npz_dir, pdb_dir)
    # AbMPNN
    # npz_dir = "/home/maghoi/repos/__personal/antifold/output/abmpnn/SAB/"
    # _ = mpnn_npzdir_to_csv(npz_dir, pdb_dir)
    """


if __name__ == "__main__":
    print(f"Converting ProteinMPNN npz files to csv files (hardcoded paths) ...")

    args = cmdline_args()
    main(args)
