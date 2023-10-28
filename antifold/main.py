import logging
import os
import sys

# import warnings
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent
sys.path.insert(0, ROOT_PATH)

from argparse import ArgumentParser, RawTextHelpFormatter

import pandas as pd

from antifold.antiscripts import (
    df_logits_to_logprobs,
    get_pdbs_logits,
    load_IF1_model,
    sample_from_df_logits,
    visualize_mutations,
    write_fasta_to_dir,
)

log = logging.getLogger(__name__)


def cmdline_args():
    # Make parser object
    usage = f"""
    # Predict on example PDBs in folder
    python antifold/main.py \
    --pdbs_csv data/example_pdbs.csv \
    --pdb_dir data/pdbs \
    --out_dir output/
    """
    p = ArgumentParser(
        description="Predict antibody variable domain, inverse folding probabilities and sample sequences with maintained fold.\nRequires IMGT-numbered PDBs with paired heavy and light chains.",
        formatter_class=RawTextHelpFormatter,
        usage=usage,
    )

    def is_valid_path(parser, arg):
        if not os.path.exists(arg):
            parser.error(f"Path {arg} does not exist!")
        else:
            return arg

    def is_valid_dir(parser, arg):
        if not os.path.isdir(arg):
            parser.error(f"Directory {arg} does not exist!")
        else:
            return arg

    p.add_argument(
        "--pdb_file",
        help="Input PDB file (for single PDB predictions)",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--heavy_chain",
        help="Ab heavy chain (for single PDB predictions)",
    )

    p.add_argument(
        "--light_chain",
        help="Ab light chain (for single PDB predictions)",
    )

    p.add_argument(
        "--pdbs_csv",
        help="Input CSV file with PDB names and H/L chains (multi-PDB predictions)",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--pdb_dir",
        help="Directory with input PDB files (multi-PDB predictions)",
        type=lambda x: is_valid_dir(p, x),
    )

    p.add_argument(
        "--out_dir",
        default="output",
        help="Output directory",
    )

    p.add_argument(
        "--regions",
        default=["CDR1", "CDR2", "CDR3"],
        type=str,
        nargs="+",
        help="Space-separated list of regions to mutate (e.g., CDR1 CDR2 CDR3H).",
    )

    p.add_argument(
        "--num_seq_per_target",
        default=20,
        type=int,
        help="Number of sequences to sample from each antibody PDB",
    )

    p.add_argument(
        "--sampling_temp",
        default=[0.2],
        type=float,
        nargs="+",
        help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.",
    )

    p.add_argument(
        "--limit_variation",
        action="store_true",
        help="Limit variation to as many mutations as expected from temperature sampling",
    )

    p.add_argument(
        "--exclude_heavy", action="store_true", help="Exclude heavy chain from sampling"
    )

    p.add_argument(
        "--exclude_light", action="store_true", help="Exclude light chain from sampling"
    )

    p.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch-size to use",
    )

    p.add_argument(
        "--num_threads",
        default=0,
        type=int,
        help="Number of CPU threads to use for parallel processing (0 = all available)",
    )

    p.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for reproducibility",
    )

    p.add_argument(
        "--model_path",
        default="models/model.pt",
        help="Output directory",
    )

    p.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbose printing",
    )

    return p.parse_args()


def sample_pdbs(
    model,
    pdbs_csv_or_dataframe,
    regions_to_mutate,
    pdb_dir="data/pdbs",
    out_dir="output/sampled",
    sample_n=10,
    sampling_temp=0.50,
    limit_expected_variation=False,
    exclude_heavy=False,
    exclude_light=False,
    batch_size=1,
    num_threads=0,
    seed=42,
    save_flag=False,
):
    # Predict with CSV on folder of solved (SAbDab) structures
    df_logits_list = get_pdbs_logits(
        model=model,
        pdbs_csv_or_dataframe=pdbs_csv_or_dataframe,
        pdb_dir=pdb_dir,
        out_dir=out_dir,
        save_flag=save_flag,
        batch_size=1,
        seed=42,
        num_threads=num_threads,
    )

    # Sample from output probabilities
    pdb_output_dict = {}
    for df_logits in df_logits_list:
        # Sample 10 sequences with a temperature of 0.50
        fasta_dict = sample_from_df_logits(
            df_logits,
            sample_n=sample_n,
            sampling_temp=sampling_temp,
            regions_to_mutate=regions_to_mutate,
            limit_expected_variation=False,
            verbose=True,
        )

        pdb_output_dict[df_logits.name] = {
            "sequences": fasta_dict,
            "logits": df_logits,
            "logprobs": df_logits_to_logprobs(df_logits),
        }

        # Write to file
        if save_flag:
            write_fasta_to_dir(fasta_dict, df_logits, out_dir=out_dir)

    return pdb_output_dict


def check_valid_input(args):
    """Checks for valid arguments"""

    # Check valid input files input arguments
    # Check either: PDB file, PDB dir or PDBs CSV inputted
    if not (args.pdb_file or (args.pdb_dir and args.pdbs_csv)):
        log.error(
            f"""Please choose one of:
        1) PDB file (--pdb_file) with --heavy_chain [letter] and --light_chain [letter]
        2) PDB directory (--pdb_dir) and CSV file (--pdbs_csv) with columns for PDB names (pdb), H (Hchain) and L (Lchain) chains
        """
        )
        sys.exit(1)

    # Option 1: PDB file, check heavy and light chain
    if args.pdb_file:
        if not (args.heavy_chain and args.light_chain):
            log.error(
                f"Single PDB input: Please specify --heavy_chain and --light_chain (e.g. --heavy_chain H --light_chain L)"
            )
            sys.exit(1)

    # Option 2: Check PDBs in PDB dir and CSV formatted correctly
    if args.pdb_dir and args.pdbs_csv:
        # Check CSV formatting
        df = pd.read_csv(args.pdbs_csv)
        if not df.columns.isin(["pdb", "Hchain", "Lchain"]).sum() >= 3:
            log.error(
                f"Multi-PDB input: Please specify CSV  with columns ['pdb', 'Hchain', 'Lchain'] with PDB names (no extension), H and L chains"
            )
            log.error(f"CSV columns: {df.columns}")
            sys.exit(1)

        # Check PDBs exist
        missing = 0
        for i, _pdb in enumerate(df["pdb"].values):
            pdb_path = f"{args.pdb_dir}/{_pdb}.pdb"
            if not os.path.exists(pdb_path):
                log.warning(
                    f"WARNING missing PDBs ({missing+1}), PDB does not exist: {pdb_path}"
                )
                missing += 1

        if missing >= 1:
            log.error(
                f"Missing {missing} PDBs specified in {args.pdbs_csv} CSV file but not found in {args.pdb_dir}"
            )
            sys.exit(1)

    # Check model exists, or set to ESM-IF1
    if not args.model_path or args.model_path == "ESM-IF1" or args.model_path == "IF1":
        log.info(
            f"Model path not specified or set to ESM-IF1. Using ESM-IF1 model instead of AntiFold fine-tuned model"
        )
        args.model_path = ""


def main(args):
    """Predicts antibody heavy and light chain inverse folding probabilities"""

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Try reading in regions
    regions_to_mutate = []
    for region in args.regions:
        try:
            positions = list(map(int, region.split(",")))
            regions_to_mutate.append(positions)
        except ValueError:
            regions_to_mutate.append(region)

    # Option 1: Single PDB
    if args.pdb_file:
        _pdb = os.path.splitext(os.path.basename(args.pdb_file))[0]
        pdbs_csv = pd.DataFrame(
            {"pdb": _pdb, "Hchain": args.heavy_chain, "Lchain": args.light_chain},
            index=[0],
        )
        pdb_dir = os.path.dirname(args.pdb_file)

    # Option 2: CSV + PDB dir
    else:
        pdbs_csv = pd.read_csv(args.pdbs_csv)
        pdb_dir = args.pdb_dir

    log.info(
        f"Will sample {args.num_seq_per_target} sequences from {len(pdbs_csv.values)} PDBs at temperature(s) {args.sampling_temp} and regions: {regions_to_mutate}"
    )

    # Load AntiFold or ESM-IF1 model
    model = load_IF1_model(args.model_path)

    # Get dict with PDBs, sampled sequences and logits / log_odds DataFrame
    pdb_output_dict = sample_pdbs(
        model=model,
        pdbs_csv_or_dataframe=pdbs_csv,
        pdb_dir=pdb_dir,
        regions_to_mutate=regions_to_mutate,
        out_dir=args.out_dir,
        sample_n=args.num_seq_per_target,
        sampling_temp=args.sampling_temp,
        limit_expected_variation=args.limit_variation,
        exclude_heavy=args.exclude_heavy,
        exclude_light=args.exclude_light,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        seed=args.seed,
        save_flag=True,
    )


if __name__ == "__main__":
    args = cmdline_args()

    # Log to file and stdout
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.abspath(f"{args.out_dir}/log.txt")

    logging.basicConfig(
        level=logging.INFO,
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

    # Check valid input
    check_valid_input(args)

    try:
        log.info(f"Sampling PDBs with Antifold ...")
        main(args)

    except Exception as E:
        log.exception(
            f"Prediction encountered an unexpected error. This is likely a bug in the server software: {E}"
        )
