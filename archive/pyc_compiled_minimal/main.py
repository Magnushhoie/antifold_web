import logging
import os
import sys
import warnings
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent
sys.path.insert(0, ROOT_PATH)

import re
from argparse import ArgumentParser, RawTextHelpFormatter

import numpy as np
import pandas as pd
import torch

import esm
from esm_util_custom import CoordBatchConverter_mask_gpu
from if1_dataset import InverseData


def cmdline_args():
    # Make parser object
    usage = f"""
    # Predict on example PDBs in folder
    python antifold/main.py \
    --pdb_csv data/example_pdbs.csv \
    --pdb_dir data/pdbs \
    --out_dir output/
    """
    p = ArgumentParser(
        description="Predict antibody heavy/light chain inverse folding probabilities",
        formatter_class=RawTextHelpFormatter,
        usage=usage,
    )

    def is_valid_path(parser, arg):
        if not os.path.exists(arg):
            parser.error(f"Path {arg} does not exist!")
        else:
            return arg

    p.add_argument(
        "--pdb_csv",
        required=True,
        help="Input CSV file with PDB names and H/L chains",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--pdb_dir",
        required=True,
        help="Directory with input PDB files",
    )

    p.add_argument(
        "--out_dir",
        default="output/",
        help="Output directory",
    )

    p.add_argument(
        "--model_path",
        default="models/model_aug23.pt",
        help="Output directory",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--batch_size",
        default=1,
        help="Batch-size to use",
    )

    p.add_argument(
        "--verbose",
        default=1,
        help="Verbose printing",
    )

    return p.parse_args()


def load_IF1_checkpoint(model, checkpoint_path: str = ""):
    # Load
    log.info(f"Loading checkpoint from {checkpoint_path}...")

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

    # Suppress regression weights warning - not needed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    if checkpoint_path:
        model = load_IF1_checkpoint(model, checkpoint_path)
    else:
        log.info(f"Loaded raw IF1 model (no checkpoint provided)")

    # Evaluation mode when predicting
    model = model.eval()

    # Send to CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.to(device)
    log.info(f"Loaded model to {device}.")

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

    alphabet = esm.data.Alphabet.from_architecture("invariant_gvp")

    mask_gap = tokens[:, 1:] != 30  # 30 is gap
    mask_pad = tokens[:, 1:] != alphabet.padding_idx  # 1 is padding
    mask_combined = mask_gap & mask_pad

    # Check that only 10x gap ("-") per sequence!
    batch_size = logits.shape[0]
    assert (mask_gap == False).sum() == batch_size * 10

    # Filter out gap (-) and padding (<pad>) positions, only keep 21x amino-acid probs (4:24) + "X"
    seqprobs_list = [logits[i, 4:25, mask_combined[i]] for i in range(len(logits))]

    return seqprobs_list


def get_dataset_dataloader(csv_pdbs, pdb_dir, batch_size):
    """Prepares dataset/dataoader from CSV file containing PDB paths and H/L chains"""

    # Load PDB coordinates
    dataset = InverseData(
        gaussian_noise_flag=False,
    )
    dataset.populate(csv_pdbs, pdb_dir)

    # Prepare torch dataloader at specified batch size
    alphabet = esm.data.Alphabet.from_architecture("invariant_gvp")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CoordBatchConverter_mask_gpu(alphabet),
    )

    return dataset, dataloader


def dataset_dataloader_to_predictions_list(model, dataset, dataloader, batch_size=1):
    """Get PDB predictions from a dataloader"""

    # Check that dataloader and dataset match, and no random shuffling
    if "random" in str(dataloader.sampler).lower():
        raise ValueError(
            "Torch DataLoader sampler must not be random. Did you forget to set torch.utils.data.DataLoader ... shuffle=False?"
        )
    if dataloader.dataset is not dataset:
        raise ValueError("Dataloader and dataset must match to align samples!")

    # Get all batch predictions
    all_seqprobs_list = []
    for bi, batch in enumerate(dataloader):
        start_index = bi * batch_size
        end_index = min(
            start_index + batch_size, len(dataset)
        )  # Adjust for the last batch
        log.info(
            f"Predicting batch {bi+1}/{len(dataloader)}: PDBs {start_index+1}-{end_index} out of {len(dataset)} total"
        )  # -1 because the end_index is exclusive

        # Test dataloader
        (
            coords,
            confidence,
            strs,
            tokens,
            padding_mask,
            loss_masks,
            res_pos,
            posins_list,
            targets,
        ) = batch

        # Test forward
        with torch.no_grad():
            prev_output_tokens = tokens[:, :-1]
            logits, extra = model.forward(  # bs x 35 x seq_len, exlude bos, eos
                coords,
                padding_mask,  # Includes masked positions
                confidence,
                prev_output_tokens,
                features_only=False,
            )

            logits = logits.detach().cpu().numpy()
            tokens = tokens.detach().cpu().numpy()

            # List of L x 21 seqprobs (20x AA, 21st == "X")
            L = logits_to_seqprobs_list(logits, tokens)
            all_seqprobs_list.extend(L)

    return all_seqprobs_list


def predictions_list_to_df_probs_list(all_seqprobs_list, dataset, dataloader):
    """PDB preds list to PDB DataFrames"""

    # Check that dataloader and dataset match, and no random shuffling
    if "random" in str(dataloader.sampler).lower():
        raise ValueError(
            "Torch DataLoader sampler must not be random. Did you forget to set torch.utils.data.DataLoader ... shuffle=False?"
        )
    if dataloader.dataset is not dataset:
        raise ValueError("Dataloader and dataset must match to align samples!")

    # Create DataFrame for each PDB
    all_df_probs_list = []
    for idx, seq_probs in enumerate(all_seqprobs_list):
        # Get PDB sequence, position+insertion code and H+L chain idxs
        (
            pdb_name,
            pdb_res,
            pdb_posins,
            pdb_chains,
        ) = get_dataset_pdb_name_res_posins_chains(dataset, idx)

        # Check matches w/ residue probs
        assert len(seq_probs) == len(pdb_posins)

        # DataFrame
        alphabet = esm.data.Alphabet.from_architecture("invariant_gvp")
        _alphabet = list("ACDEFGHIKLMNPQRSTVWYX")

        df_probs = pd.DataFrame(
            data=seq_probs,
            columns=alphabet.all_toks[4:25],
        )
        df_probs = df_probs[_alphabet]

        df_probs.insert(0, "aa_orig", pdb_res)
        df_probs.insert(
            0,
            "aa_pred",
            [
                list(_alphabet)[i]
                for i in df_probs[list(_alphabet)].values.argmax(axis=1)
            ],
        )

        df_probs.insert(2, "pdb_res", pdb_res)
        df_probs.insert(3, "pdb_posins", pdb_posins)
        df_probs.insert(4, "pdb_chain", pdb_chains)
        df_probs.name = pdb_name
        all_df_probs_list.append(df_probs)

    return all_df_probs_list


def df_probs_list_to_csvs(df_probs_list, out_dir):
    """Save df_probs_list to CSVs"""
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"Saving {len(df_probs_list)} CSVs to {out_dir}")

    for df in df_probs_list:
        outpath = f"{out_dir}/{df.name}.csv"
        log.info(f"Writing predictions for {df.name} to {outpath}")
        df.to_csv(outpath)


def predict_and_save(model, csv_pdbs, pdb_dir, out_dir, batch_size=1, save_flag=True):
    """Predict PDBs from a CSV file"""

    log.info(f"\nPredicting PDBs from CSV file: {csv_pdbs}")

    if save_flag:
        log.info(f"Saving prediction CSVs to {out_dir}")

    # Load PDBs
    dataset, dataloader = get_dataset_dataloader(
        csv_pdbs, pdb_dir, batch_size=batch_size
    )

    # Predict PDBs -> df_probs
    predictions_list = dataset_dataloader_to_predictions_list(
        model, dataset, dataloader, batch_size=batch_size
    )
    df_probs_list = predictions_list_to_df_probs_list(
        predictions_list, dataset, dataloader
    )

    # Save df_probs to CSVs
    if save_flag:
        df_probs_list_to_csvs(df_probs_list, out_dir)

    return df_probs_list


def main(args):
    """Predicts AbMPNN and IF1-raw models on Abmpnn test set (SAbDab and ImmuneBuilder versions)"""

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = load_IF1_model(args.model_path)

    # Predict PDBs listed in CSV file
    _ = predict_and_save(
        model, args.pdb_csv, args.pdb_dir, args.out_dir, args.batch_size
    )


if __name__ == "__main__":
    args = cmdline_args()

    # Log to file and stdout
    # If verbose == 0, only errors are printed (default 1)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.abspath(f"{args.out_dir}/log.txt")

    logging.basicConfig(
        level=logging.ERROR,
        format="[{asctime}] {message}",
        style="{",
        handlers=[
            logging.FileHandler(filename=log_path, mode="w"),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    # INFO prints total summary and errors (default)
    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)

    # DEBUG prints every major step
    elif args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info(f"Predicting PDBs with Antifold ...")
    main(args)
