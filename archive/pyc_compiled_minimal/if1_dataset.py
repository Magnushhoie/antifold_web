import logging

logging.basicConfig(level=logging.INFO, format="[{asctime}] {message}", style="{")
log = logging.getLogger(__name__)

import sys

sys.path.insert(0, ".")

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import esm
from esm_multichain_util_custom import (concatenate_coords_HL,
                                                 load_complex_coords)
from esm_util_custom import CoordBatchConverter_mask


class InverseData(torch.utils.data.Dataset):
    """
    Prepare dataset for ESM-IF1, including span masking and adding gaussian noise, returning
        batches with coords, confidence, strs, tokens, padding_mask
    Returns: coords, confidence, seq, res_pos, loss_mask, targets
    """

    def __init__(
        self,
        verbose: int = 0,
        loss_mask_flag: bool = False,
        gaussian_noise_flag: bool = True,
        gaussian_scale_A: float = 0.1,
    ):
        # Variables
        self.loss_mask_flag = loss_mask_flag
        self.gaussian_noise_flag = gaussian_noise_flag
        self.gaussian_scale_A = gaussian_scale_A
        self.seq_pdb_fasta_mismatches = 0
        self.verbose = verbose

    def load_coords_HL(
        self, pdb_path: str, Hchain: str, Lchain: str
    ) -> Tuple[np.array, str, np.array]:
        """Read pdb file and extract coordinates of backbone (N, CA, C) atoms of given chains
        Args:
            file path: path to pdb file
            Hchain: heavy chain
            Lchain: light chain
        Returns:
            coords_concatenated: 3d backbone coordinates of extracted structure w/ padding - padded coords set to inf
            seq_concatenated: AA sequence w/ padding
            pos_concatenated: residue positions w/ padding
        """

        # coords, seq = esm.inverse_folding.util.load_coords(fpath=pdb_path, chain=chain)
        coords_dict, seq_dict, pos_dict, posins_dict = load_complex_coords(
            pdb_path, [Hchain, Lchain]
        )
        (
            coords_concatenated,
            seq_concatenated,
            pos_concatenated,
            posins_concatenated,
        ) = concatenate_coords_HL(
            coords_dict, seq_dict, pos_dict, posins_dict, heavy_chain_id=Hchain
        )

        return (
            coords_concatenated,
            seq_concatenated,
            pos_concatenated,
            posins_concatenated,
        )

    def add_gaussian_noise(self, coords: np.array, scale=0.1):
        """Add Gaussian noise at scale 0.1A to each coordinate
        Args:
            coords: 3d backbone coordinates of full_structure (as formatted by load_coords, can be span-masked)
        Returns:
            coords with Gaussian noise added
        """
        return coords + np.random.normal(scale=scale, size=coords.shape)

    def populate(self, pdb_csv: str, pdb_dir: str, verbose: int = 1):
        """
        Gets the actual PDB paths to be used for training and testing,
        will filter on the PDBs present in the paragraph CSV dict if set.

        Args:
            pdb_csv: path to csv file containing pdb, Hchain and Lchain
        """

        if not os.path.exists(pdb_csv):
            log.error(f"Unable to find pdb_csv {pdb_csv}")
            sys.exit(1)

        # Load CSV and check it
        df = pd.read_csv(pdb_csv)

        if not len(df) >= 2:
            log.error(f"CSV file {pdb_csv} must contain at least 1 PDB")
            sys.exit(1)

        if not df.columns.isin(["pdb", "Hchain", "Lchain"]).sum() == 3:
            log.error(
                f"CSV file requires columns 'pdb, Hchain, Lchain': found {df.columns}"
            )
            sys.exit(1)

        if verbose:
            log.info(f"Populating {len(df)} PDBs from {pdb_csv}")

        # Create list of PDB paths and check that they exist
        pdb_path_list = []
        for _pdb in df["pdb"]:
            pdb_path = f"{pdb_dir}/{_pdb}.pdb"
            pdb_path_list.append(pdb_path)

            if not os.path.exists(pdb_path):
                raise Exception(f"Unable to find PDB file: {pdb_path}")

        # PDB paths
        self.pdb_paths = pdb_path_list

        # PDB info dict
        df.index = pdb_path_list
        df["pdb_path"] = pdb_path_list
        pdb_info_dict = df.to_dict("index")
        self.pdb_info_dict = pdb_info_dict

    def __getitem__(self, idx: int):
        """
        Format data to pass to PyTorch DataLoader (with collate_fn = util.CoordBatchConverter)
        """

        # obtain pdb_info for entry with index idx - pdb_info contains the pdb_path (generated in populate)
        pdb_path = self.pdb_paths[idx]
        pdb_info = self.pdb_info_dict[pdb_path]
        _pdb = pdb_info["pdb"] + "_" + pdb_info["Hchain"] + pdb_info["Lchain"]

        coords, seq_pdb, pos_pdb, pos_pdb_arr_str = self.load_coords_HL(
            pdb_path, pdb_info["Hchain"], pdb_info["Lchain"]
        )

        if self.verbose >= 2:
            print(
                f"""
            Loaded coords for {_pdb}, shape {coords.shape}
            seq_pdb len {len(seq_pdb)},
            pos_pdb {pos_pdb.shape},
            pos_pdb_arr_str {pos_pdb_arr_str.shape}
            from {pdb_path} (Hchain {pdb_info['Hchain']}, Lchain {pdb_info['Lchain']})
            """
            )

        # If no CSV provided, fill in empty (un-used) targets shape of PDB
        else:
            targets = np.full(len(pos_pdb_arr_str), np.nan)

        # Add (0.1 Å) gaussian noise to Ca, C, N co-ordinates
        if self.gaussian_noise_flag:
            coords = self.add_gaussian_noise(coords=coords, scale=self.gaussian_scale_A)

        # Initialize empty loss_mask
        loss_mask = np.full(len(coords), fill_value=False)

        # reset padding coords between H and L to np.nan (instead of np.inf, which was set so as not to interfere with masking)
        coords[pos_pdb_arr_str == "nan"] = np.nan

        # confidence currently not taken - update
        confidence = None

        # check masking done correctly - no masking in linker region (between heavy and light chains)
        assert loss_mask[np.where(pos_pdb_arr_str == "nan")].sum() == 0

        return coords, confidence, seq_pdb, pos_pdb, pos_pdb_arr_str, loss_mask, targets

    def __len__(self):
        """Get number of entries in dataset"""
        return len(self.pdb_info_dict)


if __name__ == "__main__":
    print(f"Testing dataset ...")

    # Hyperparams
    batch_size = 2

    # PDBs to load
    csv_pdbs = "data/example_pdbs.csv"
    dataset = InverseData()

    # Load actual PDBs, filtered on Paragraph
    dataset.populate(pdb_csv=csv_pdbs, pdb_dir="data/pdbs", verbose=2)

    print("Test 1: Dataset")
    coords, confidence, seq, res_pos, posins_list, loss_mask, targets = next(
        iter(dataset)
    )

    # coords, confidence, seq, res_pos, loss_mask, targets
    print(f"coords {coords.shape}: {coords.shape}")
    print(f"seq {len(seq)}: {seq}")
    print(f"res_pos {res_pos.shape}: {res_pos}")
    print(f"posins_list {len(posins_list)}: {posins_list}")
    print(f"loss_mask {loss_mask.sum()}/{loss_mask.shape}: {loss_mask}")
    print(f"targets {targets.shape}: {targets}")
    print(f"targets positions: {res_pos[targets.astype(bool)]}")
    print(
        f"Masked positions {loss_mask.sum()}/{len(loss_mask)}: {np.array(res_pos)[loss_mask]}"
    )

    alphabet = esm.data.Alphabet.from_architecture("invariant_gvp")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # if True, data shuffled at every epoch
        collate_fn=CoordBatchConverter_mask(alphabet),
    )

    input("Test 2: Dataloader. Continue?")
    (
        coords,
        confidence,
        strs,
        tokens,
        padding_mask,
        loss_mask,
        res_pos,
        posins_list,
        targets,
    ) = next(iter(dataloader))
    print(f"coords: {coords.shape}, {coords.dtype}")
    print(f"seq: {seq}")
    print(f"res_pos: {res_pos}")
    print(f"posins_list 0: {posins_list[0]}")
    print(f"loss_mask: {loss_mask}")
    print(f"targets: {targets}")

    print(f"Done!")