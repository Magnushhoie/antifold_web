import os
import re

import numpy as np
import torch
import antifold.esm


def load_IF1_checkpoint(model, checkpoint_path: str = ""):
    # Load
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Check for CPU/GPU load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device(device))

    # PYL checkpoint ?
    if "model_state_dict" in checkpoint_dict.keys():
        pretrained_dict = {
            re.sub("model.", "", k): v
            for k, v in checkpoint_dict["model_state_dict"].items()
        }

    # PYL checkpoint ?
    elif "state_dict" in checkpoint_dict.keys():
        # Remove "model." from keys
        pretrained_dict = {
            re.sub("model.", "", k): v for k, v in checkpoint_dict["state_dict"].items()
        }

    # IF1-raw?
    else:
        pretrained_dict = checkpoint_dict

    # Load pretrained weights
    model.load_state_dict(pretrained_dict)

    return model


def load_IF1_model(checkpoint_path: str = ""):
    """Load raw/FT IF1 model"""

    model, _ = antifold.esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    if checkpoint_path:
        model = load_IF1_checkpoint(model, checkpoint_path)

    # Eval and send to device
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.to(device)
    print(f"Loaded model to {device}.")

    return model


def get_dataset_pdb_name_res_posins_chains(dataset, idx):
    """Gets PDB sequence, position+insertion codes and chains from dataset"""

    # Get PDB sequence, position+insertion codes and chains from dataset
    pdb_path = dataset.pdb_paths[idx]
    pdb_info = dataset.pdb_info_dict[pdb_path]
    pdb_name = os.path.splitext(os.path.basename(pdb_info["pdb_path"]))[0]

    # Sequence - gaps
    seq = dataset[idx][2]
    pdb_res = [aa for aa in seq if aa != "-"]

    # Position + insertion codes - gaps
    posins = dataset[idx][4]
    pdb_posins = [p for p in posins if p != "nan"]

    # PDB chains (Always H + L), can infer L idxs from L length
    pdb_chains = np.array([pdb_info["Hchain"]] * len(pdb_posins))
    L_length = len(posins) - np.where(np.array(posins) == "nan")[0].max()
    pdb_chains[-L_length:] = pdb_info["Lchain"]

    return pdb_name, pdb_res, pdb_posins, pdb_chains


def logits_to_seqprobs_list(logits, tokens):
    """Convert logits (bs x 35 x L) ot list of L x 20 seqprobs"""

    alphabet = antifold.esm.data.Alphabet.from_architecture("invariant_gvp")

    mask_gap = tokens[:, 1:] != 30  # 30 is gap
    mask_pad = tokens[:, 1:] != alphabet.padding_idx  # 1 is padding
    mask_combined = mask_gap & mask_pad

    # Check that only 10x gap ("-") per sequence!
    batch_size = logits.shape[0]
    assert (mask_gap == False).sum() == batch_size * 10

    # Filter out gap (-) and padding (<pad>) positions, only keep 21x amino-acid probs (4:24) + "X"
    seqprobs_list = [logits[i, 4:25, mask_combined[i]] for i in range(len(logits))]

    return seqprobs_list
