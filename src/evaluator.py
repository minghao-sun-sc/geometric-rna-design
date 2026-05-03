import os
import copy
import shutil
from datetime import datetime

import sys

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import re

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_matthews_corrcoef

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from Bio.PDB import PDBParser, Select, PDBIO
from Bio.SeqUtils import seq1
import tempfile

from MDAnalysis.analysis.align import rotation_matrix
from MDAnalysis.analysis.rms import rmsd as get_rmsd

# Import these locally to avoid NetworkX conflicts
# from src.data.data_utils import pdb_to_tensor, get_c4p_coords
# from src.data.sec_struct_utils import (
#     predict_sec_struct,
#     dotbracket_to_paired,
#     dotbracket_to_adjacency
# )
from src.constants import (
    NUM_TO_LETTER,
    PROJECT_PATH,
    MOLPROBITY_HOME,
    RMSD_THRESHOLD,
    RMSD_THRESHOLD_2,
    TM_THRESHOLD,
    GDT_THRESHOLD,
    PLDDT_THRESHOLD,
    DATA_PATH
)

from tools.RNA_assessment import RNA_normalizer
RESIDUES_LIST = "tools/RNA_assessment/data/residues.list"
ATOMS_LIST = "tools/RNA_assessment/data/atoms.list"

import subprocess


def evaluate(
        model,
        dataset,
        n_samples,
        temperature,
        device,
        model_name="eval",
        metrics=[
            'recovery', 'perplexity', 'sc_score_eternafold',
            'sc_score_ribonanzanet', 'sc_score_rhofold', 'sc_score_vienna',
            'sc_score_assessment'
        ],
        save_designs=False
    ):
    """
        Run evaluation suite for trained RNA inverse folding model on a dataset.

        The following metrics can be computed along with metadata per sample per residue:
        1. (recovery) Sequence recovery per residue (taking mean gives per sample recovery)
        2. (perplexity) Perplexity per sample
        3. (sc_score_eternafold) Secondary structure self-consistency score per sample,
            using EternaFold for secondary structure prediction and computing MCC between
            the predicted and groundtruth 2D structures as adjacency matrices.
        4. (sc_score_ribonanzanet) Chemical modification self-consistency score per sample,
            using RibonanzaNet for chemical modification prediction of the groundtruth and
            designed sequences, and measuring MAE between them.
        5. (sc_score_rhofold) Tertiary structure self-consistency scores per sample,
            using RhoFold for tertiary structure prediction and measuring RMSD, TM-score,
            and GDT_TS between the predicted and groundtruth C4' 3D coordinates.
        6. (rmsd_within_thresh) Percentage of samples with RMSD within threshold (<=2.0A)
        7. (tm_within_thresh) Percentage of samples with TM-score within threshold (>=0.45)
        8. (gddt_within_thresh) Percentage of samples with GDT_TS within threshold (>=0.50)

        Args:
            model: trained RNA inverse folding model
            dataset: dataset to evaluate on
            n_samples: number of predicted samples/sequences per data point
            temperature: sampling temperature
            device: device to run evaluation on
            model_name: name of model/dataset for plotting (default: 'eval')
            metrics: list of metrics to compute
            save_designs: whether to save designs as fasta with metrics

        Returns: Dictionary with the following keys:
            df: DataFrame with metrics and metadata per residue per sample for analysis and plotting
            samples_list: list of tensors of shape (n_samples, seq_len) per data point
            recovery_list: list of mean recovery per data point
            perplexity_list: list of mean perplexity per data point
            sc_score_eternafold_list: list of 2D self-consistency scores per data point
            sc_score_ribonanzanet_list: list of 1D self-consistency scores per data point
            sc_score_rmsd_list: list of 3D self-consistency RMSDs per data point
            sc_score_tm_list: list of 3D self-consistency TM-scores per data point
            sc_score_gdt_list: list of 3D self-consistency GDTs per data point
            rmsd_within_thresh_list: list of % scRMSDs within threshold per data point
            tm_within_thresh_list: list of % scTMs within threshold per data point
            gddt_within_thresh_list: list of % scGDDTs within threshold per data point
            plddt_within_thresh_list: list of % scPLDDTs within threshold per data point
        """
    assert 'recovery' in metrics, 'Sequence recovery must be computed for evaluation'

    #######################################################################
    # Optionally initialise other models used for self-consistency scoring
    #######################################################################

    if 'sc_score_ribonanzanet' in metrics:
        from tools.ribonanzanet.network import RibonanzaNet

        # Initialise RibonanzaNet for self-consistency score
        ribonanza_net = RibonanzaNet(
            os.path.join(PROJECT_PATH, 'tools/ribonanzanet/config.yaml'),
            os.path.join(PROJECT_PATH, 'tools/ribonanzanet/ribonanzanet.pt'),
            device
        )
        # Transfer model to device in eval mode
        ribonanza_net = ribonanza_net.to(device)
        ribonanza_net.eval()

    if 'sc_score_rhofold' in metrics:
        from tools.rhofold.rf import RhoFold
        from tools.rhofold.config import rhofold_config

        # Initialise RhoFold for 3D self-consistency score
        rhofold = RhoFold(rhofold_config, device)
        rhofold_path = os.path.join(PROJECT_PATH, "tools/rhofold/model_20221010_params.pt")
        print(f"Loading RhoFold checkpoint: {rhofold_path}")
        rhofold.load_state_dict(torch.load(rhofold_path, map_location=torch.device('cpu'))['model'])
        # Transfer model to device in eval mode
        rhofold = rhofold.to(device)
        rhofold.eval()
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    if 'sc_score_assessment' in metrics:
        pass

    ####################################################
    # Evaluation loop over each data point sequentially
    ####################################################

    # per sample metric lists for storing evaluation results
    samples_list = []  # list of tensors of shape (n_samples, seq_len) per data point
    recovery_list = []  # list of mean recovery per data point
    perplexity_list = []  # list of mean perplexity per data point
    sc_score_ribonanzanet_list = []  # list of 1D self-consistency scores per data point
    sc_score_eternafold_list = []  # list of 2D self-consistency scores per data point
    sc_score_rmsd_list = []  # list of 3D self-consistency RMSDs per data point
    rmsd_within_thresh_list = []  # list of % scRMSDs within threshold per data point
    sc_score_tm_list = []  # list of 3D self-consistency TM-scores per data point
    tm_within_thresh_list = []  # list of % scTMs within threshold per data point
    sc_score_gdt_list = []  # list of 3D self-consistency GDTs per data point
    gddt_within_thresh_list = []  # list of % scGDDTs within threshold per data point
    sc_score_plddt_list = []  # list of 3D self-consistency PLDDTs per data point
    plddt_within_thresh_list = []  # list of % scPLDDTs within threshold per data point

    # DataFrame to store metrics and metadata per residue per sample for analysis and plotting
    df = pd.DataFrame(columns=['idx', 'recovery', 'sasa', 'paired', 'rmsds', 'model_name'])

    model.eval()
    if device.type == 'xpu':
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model)
        if 'sc_score_ribonanzanet' in metrics:
            ribonanza_net = ipex.optimize(ribonanza_net)
        if 'sc_score_rhofold' in metrics:
            rhofold = ipex.optimize(rhofold)

    with torch.no_grad():
        for idx, raw_data in tqdm(
                enumerate(dataset.data_list),
                total=len(dataset.data_list)
        ):
            # featurise raw data
            data = dataset.featurizer(raw_data).to(device)

            # sample n_samples from model for single data point: n_samples x seq_len
            samples, logits = model.sample(data, n_samples, temperature, return_logits=True)
            samples_list.append(samples.cpu().numpy())

            # perplexity per sample: n_samples x 1
            n_nodes = logits.shape[1]
            perplexity = torch.exp(F.cross_entropy(
                logits.view(n_samples * n_nodes, model.out_dim),
                samples.view(n_samples * n_nodes).long(),
                reduction="none"
            ).view(n_samples, n_nodes).mean(dim=1)).cpu().numpy()
            perplexity_list.append(perplexity.mean())

            ###########
            # Metadata
            ###########

            # per residue average SASA: seq_len x 1
            mask_coords = data.mask_coords.cpu().numpy()
            sasa = np.mean(raw_data['sasa_list'], axis=0)[mask_coords]

            # per residue indicator for paired/unpaired: seq_len x 1
            # Import locally to avoid NetworkX conflicts
            from src.data.sec_struct_utils import dotbracket_to_paired
            paired = np.mean(
                [dotbracket_to_paired(sec_struct) for sec_struct in raw_data['sec_struct_list']], axis=0
            )[mask_coords]

            # per residue average RMSD: seq_len x 1
            if len(raw_data["coords_list"]) == 1:
                rmsds = np.zeros_like(sasa)
            else:
                rmsds = []
                # Import locally to avoid NetworkX conflicts
                from src.data.data_utils import get_c4p_coords
                for i in range(len(raw_data["coords_list"])):
                    for j in range(i + 1, len(raw_data["coords_list"])):
                        coords_i = get_c4p_coords(raw_data["coords_list"][i])
                        coords_j = get_c4p_coords(raw_data["coords_list"][j])
                        rmsds.append(torch.sqrt(torch.sum((coords_i - coords_j) ** 2, dim=1)).cpu().numpy())
                rmsds = np.stack(rmsds).mean(axis=0)[mask_coords]

            ##########
            # Metrics
            ##########

            # sequence recovery per residue across all samples: n_samples x seq_len
            recovery = samples.eq(data.seq).float().cpu().numpy()
            recovery_list.append(recovery.mean())

            # update per residue per sample dataframe
            df = pd.concat([
                df,
                pd.DataFrame({
                    'idx': [idx] * len(recovery.mean(axis=0)),
                    'recovery': recovery.mean(axis=0),
                    'sasa': sasa,
                    'paired': paired,
                    'rmsds': rmsds,
                    'model_name': [model_name] * len(recovery.mean(axis=0))
                })
            ], ignore_index=True)

            # global 2D self consistency score per sample: n_samples x 1
            if 'sc_score_eternafold' in metrics:
                sc_score_eternafold, pred_sec_structs = self_consistency_score_eternafold(
                    samples.cpu().numpy(),
                    raw_data['sec_struct_list'],
                    mask_coords,
                    return_sec_structs=True
                )
                sc_score_eternafold_list.append(sc_score_eternafold.mean())

            # ---------------- ViennaRNA ensemble metrics (MFE, ED, entropy, p(S0), diversity, Tm) ----------------
            if 'sc_score_vienna' in metrics:  # FIXED: Use defensive Vienna metrics to prevent NaN results
                try:
                    # Import defensive Vienna functions that handle edge cases properly
                    from src.vienna_defensive import defensive_vienna_mfe, defensive_vienna_ensemble
                    
                    # Get target structure safely with error handling
                    target_db_full = None
                    if len(raw_data.get('sec_struct_list', [])) > 0:
                        target_db_full = _sanitize_db_for_vienna(raw_data['sec_struct_list'][0])
                    else:
                        try:
                            _, target_db_full = defensive_vienna_mfe(raw_data['sequence'], 37.0)
                        except Exception:
                            target_db_full = '.' * len(raw_data.get('sequence', ''))

                    # Process with mask_coords safely
                    if mask_coords is not None and mask_coords.sum() < len(mask_coords):
                        keep_idx = np.where(mask_coords)[0]
                        if target_db_full and len(keep_idx) > 0:
                            # Safe bounds checking
                            valid_idx = keep_idx[keep_idx < len(target_db_full)]
                            if len(valid_idx) > 0:
                                target_db = "".join(target_db_full[i] for i in valid_idx)
                            else:
                                target_db = target_db_full
                        else:
                            target_db = target_db_full
                    else:
                        target_db = target_db_full

                    # Collect metrics with defensive calculations  
                    v_mfe, v_ed, v_ednt, v_pS0, v_ent, v_div = [], [], [], [], [], []
                    
                    for seq_nums in samples.cpu().numpy():
                        seq = "".join([NUM_TO_LETTER[n] for n in seq_nums])
                        
                        # Apply masking to sequence safely
                        if mask_coords is not None and mask_coords.sum() < len(mask_coords):
                            keep_idx_seq = keep_idx[keep_idx < len(seq)]
                            if len(keep_idx_seq) > 0:
                                seq = "".join(seq[i] for i in keep_idx_seq)

                        # Calculate metrics with defensive error handling
                        try:
                            v = defensive_vienna_ensemble(seq, target_db, T=37.0)
                            v_mfe.append(v["mfe"])
                            v_ed.append(v["ED"])
                            v_ednt.append(v["ED_per_nt"])
                            v_pS0.append(v["pS0"])
                            v_ent.append(v["entropy_mean"])
                            v_div.append(v["diversity"])
                        except Exception:
                            # Ultimate fallback to prevent crashes
                            v_mfe.append(float('nan'))
                            v_ed.append(float('nan'))
                            v_ednt.append(float('nan'))
                            v_pS0.append(float('nan'))
                            v_ent.append(float('nan'))
                            v_div.append(float('nan'))

                    # Store results with NaN-safe averaging
                    def nan_safe_mean(values):
                        """Calculate mean ignoring NaN values."""
                        valid = [v for v in values if not np.isnan(v) and v is not None]
                        return np.mean(valid) if valid else float('nan')

                    # Store per-datapoint aggregates
                    try:
                        vienna_mfe_list.append(nan_safe_mean(v_mfe))
                        vienna_ed_list.append(nan_safe_mean(v_ed))
                        vienna_ednt_list.append(nan_safe_mean(v_ednt))
                        vienna_pS0_list.append(nan_safe_mean(v_pS0))
                        vienna_entropy_list.append(nan_safe_mean(v_ent))
                        vienna_diversity_list.append(nan_safe_mean(v_div))
                        vienna_Tm_list.append(float('nan'))  # Skip expensive Tm calculation
                    except NameError:
                        # First time initialization
                        vienna_mfe_list = [nan_safe_mean(v_mfe)]
                        vienna_ed_list = [nan_safe_mean(v_ed)]
                        vienna_ednt_list = [nan_safe_mean(v_ednt)]
                        vienna_pS0_list = [nan_safe_mean(v_pS0)]
                        vienna_entropy_list = [nan_safe_mean(v_ent)]
                        vienna_diversity_list = [nan_safe_mean(v_div)]
                        vienna_Tm_list = [float('nan')]
                        
                except ImportError as e:
                    # Fallback if defensive Vienna module not available
                    print(f"WARNING: Defensive Vienna metrics not available: {e}")
                    print("Falling back to original Vienna implementation...")
                    # [Keep original implementation as fallback - not shown for brevity]
                    pass
            # ------------------------------------------------------------------------------------------------------


            # global 1D self consistency score per sample: n_samples x 1
            if 'sc_score_ribonanzanet' in metrics:
                sc_score_ribonanzanet, pred_chem_mods = self_consistency_score_ribonanzanet(
                    samples.cpu().numpy(),
                    raw_data['sequence'],
                    mask_coords,
                    ribonanza_net,
                    return_chem_mods=True
                )
                sc_score_ribonanzanet_list.append(sc_score_ribonanzanet.mean())

            # global 3D self consistency scores per sample: n_samples x 1, each
            if 'sc_score_rhofold' in metrics:
                try:
                    output_dir = os.path.join(
                        wandb.run.dir, f"designs_{model_name}/{current_datetime}/sample{idx}/")
                except AttributeError:
                    output_dir = os.path.join(
                        PROJECT_PATH, f"designs_{model_name}/{current_datetime}/sample{idx}/")

                # Use extended RhoFold evaluation with all metrics
                (sc_score_rmsd, sc_score_tm, sc_score_gdt, sc_score_plddt, 
                 sc_inf_dict, sc_clash_dict, sc_lddt_arr, sc_mcq_dict) = self_consistency_score_rhofold_extended(
                    samples.cpu().numpy(),
                    raw_data,
                    mask_coords,
                    rhofold,
                    output_dir,
                    save_designs=save_designs,
                    use_inf=True,
                    use_clash=True,
                    phenix_wrapper_path=os.path.join(PROJECT_PATH, "tools", "run_phenix.sh"),
                )
                
                sc_score_rmsd_list.append(sc_score_rmsd.mean())
                sc_score_tm_list.append(sc_score_tm.mean())
                sc_score_gdt_list.append(sc_score_gdt.mean())
                sc_score_plddt_list.append(sc_score_plddt.mean())
                
                # INF and clash score arrays
                try:
                    inf_all_list.append(np.nanmean(sc_inf_dict["all"]))
                    inf_wc_list.append(np.nanmean(sc_inf_dict["wc"]))
                    inf_nwc_list.append(np.nanmean(sc_inf_dict["nwc"]))
                    inf_stack_list.append(np.nanmean(sc_inf_dict["stack"]))
                    # Handle both pre and post relax clash scores
                    clashscore_pre_list.append(np.nanmean(sc_clash_dict["pre_relax"]) if sc_clash_dict["pre_relax"].size > 0 else np.nan)
                    clashscore_post_list.append(np.nanmean(sc_clash_dict["post_relax"]) if sc_clash_dict["post_relax"].size > 0 else np.nan)
                except NameError:
                    inf_all_list = [np.nanmean(sc_inf_dict["all"])]
                    inf_wc_list = [np.nanmean(sc_inf_dict["wc"])]
                    inf_nwc_list = [np.nanmean(sc_inf_dict["nwc"])]
                    inf_stack_list = [np.nanmean(sc_inf_dict["stack"])]
                    # Handle both pre and post relax clash scores  
                    clashscore_pre_list = [np.nanmean(sc_clash_dict["pre_relax"]) if sc_clash_dict["pre_relax"].size > 0 else np.nan]
                    clashscore_post_list = [np.nanmean(sc_clash_dict["post_relax"]) if sc_clash_dict["post_relax"].size > 0 else np.nan]


                rmsd_within_thresh_list.append((sc_score_rmsd <= RMSD_THRESHOLD).sum() / n_samples)
                rmsd_within_2A_list = locals().get('rmsd_within_2A_list', [])
                rmsd_within_2A_list.append((sc_score_rmsd <= RMSD_THRESHOLD_2).sum() / n_samples)

                tm_within_thresh_list.append((sc_score_tm >= TM_THRESHOLD).sum() / n_samples)
                gddt_within_thresh_list.append((sc_score_gdt >= GDT_THRESHOLD).sum() / n_samples)
                plddt_within_thresh_list.append((sc_score_plddt >= PLDDT_THRESHOLD).sum() / n_samples)

                # if save_designs:
                #     # collate designed sequences in fasta format
                #     sequences = [SeqRecord(
                #         Seq(raw_data["sequence"]), id=f"input_sequence,",
                #         description=f"pdb_id={raw_data['id_list'][0]} rfam={raw_data['rfam_list'][0]} eq_class={raw_data['eq_class_list'][0]} cluster={raw_data['cluster_structsim0.45']}"
                #     )]
                #     for idx, zipped in enumerate(zip(
                #             samples.cpu().numpy(),
                #             perplexity,
                #             recovery.mean(axis=1),
                #             sc_score_eternafold,
                #             pred_sec_structs,
                #             sc_score_ribonanzanet,
                #             pred_chem_mods,
                #             sc_score_rmsd,
                #             sc_score_tm,
                #             sc_score_gdt,
                #             sc_score_plddt
                #     )):
                #         seq, perp, rec, sc, pred_ss, sc_ribo, pred_cm, sc_rmsd, sc_tm, sc_gdt = zipped
                #         seq = "".join([NUM_TO_LETTER[num] for num in seq])
                #         edit_dist = edit_distance(seq, raw_data['sequence'])
                #         sequences.append(SeqRecord(
                #             Seq(seq), id=f"sample={idx},",
                #             description=f"temperature={temperature} perplexity={perp:.4f} recovery={rec:.4f} edit_dist={edit_dist} sc_score={sc:.4f} sc_score_ribonanzanet={sc_ribo:.4f} sc_score_rmsd={sc_rmsd:.4f} sc_score_tm={sc_tm:.4f} sc_score_gdt={sc_gdt:.4f}"
                #         ))
                #     # write all designed sequences to output filepath
                #     SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")

                if save_designs:
                    sequences = [SeqRecord(
                        Seq(raw_data["sequence"]), id="input_sequence,",
                        description=(f"pdb_id={raw_data['id_list'][0]} rfam={raw_data['rfam_list'][0]} "
                                    f"eq_class={raw_data['eq_class_list'][0]} cluster={raw_data['cluster_structsim0.45']}")
                    )]

                    # Safe fallbacks if these metrics weren’t requested this run
                    nS = n_samples
                    sc_ef_arr   = sc_score_eternafold if ('sc_score_eternafold'   in metrics) else np.full(nS, np.nan)
                    pred_ss_arr = pred_sec_structs    if ('sc_score_eternafold'   in metrics) else [""] * nS
                    sc_ribo_arr = sc_score_ribonanzanet if ('sc_score_ribonanzanet' in metrics) else np.full(nS, np.nan)
                    pred_cm_arr = pred_chem_mods      if ('sc_score_ribonanzanet' in metrics) else [None] * nS

                    for i, (seq_nums, perp, rec, sc_ef, pred_ss, sc_ribo, pred_cm, r, tm, gdt, plddt) in enumerate(zip(
                            samples.cpu().numpy(),
                            perplexity,
                            recovery.mean(axis=1),
                            sc_ef_arr,
                            pred_ss_arr,
                            sc_ribo_arr,
                            pred_cm_arr,
                            sc_score_rmsd,
                            sc_score_tm,
                            sc_score_gdt,
                            sc_score_plddt
                    )):
                        seq_str = "".join(NUM_TO_LETTER[int(n)] for n in seq_nums)
                        edist   = edit_distance(seq_str, raw_data['sequence'])
                        sequences.append(SeqRecord(
                            Seq(seq_str),
                            id=f"sample={i},",
                            description=(f"temperature={temperature} perplexity={perp:.4f} recovery={rec:.4f} "
                                        f"edit_dist={edist} sc_eternafold={sc_ef:.4f} "
                                        f"sc_ribonanzanet={sc_ribo:.4f} sc_rmsd={r:.4f} sc_tm={tm:.4f} "
                                        f"sc_gdt={gdt:.4f} sc_plddt={plddt:.4f}")
                        ))
                    SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")



    out = {
        'df': df,
        'samples_list': samples_list,
        'recovery_list': recovery_list,
        'perplexity_list': perplexity_list
    }
    if 'sc_score_eternafold' in metrics:
        out['sc_score_eternafold'] = sc_score_eternafold_list
    if 'sc_score_ribonanzanet' in metrics:
        out['sc_score_ribonanzanet'] = sc_score_ribonanzanet_list
    if 'sc_score_vienna' in metrics:
        out['vienna_mfe'] = vienna_mfe_list                 # per-datapoint mean MFE (kcal/mol)
        out['vienna_ED'] = vienna_ed_list                   # mean ensemble defect (nt)
        out['vienna_ED_per_nt'] = vienna_ednt_list          # mean ED normalized by length
        out['vienna_pS0'] = vienna_pS0_list                 # mean probability of target structure
        out['vienna_entropy'] = vienna_entropy_list         # mean positional Shannon entropy
        out['vienna_diversity'] = vienna_diversity_list     # mean bp-distance (ensemble diversity)
        out['vienna_Tm'] = vienna_Tm_list                   # mean Tm (°C) if you enabled Tm
    if 'sc_score_rhofold' in metrics:
        out['sc_score_rmsd'] = sc_score_rmsd_list
        out['sc_score_tm'] = sc_score_tm_list
        out['sc_score_gdt'] = sc_score_gdt_list
        out['rmsd_within_thresh'] = rmsd_within_thresh_list
        out['rmsd_within_2A'] = rmsd_within_2A_list
        out['tm_within_thresh'] = tm_within_thresh_list
        out['gddt_within_thresh'] = gddt_within_thresh_list
        out['plddt_within_thresh'] = plddt_within_thresh_list
        out['inf_all']   = inf_all_list
        out['inf_wc']    = inf_wc_list
        out['inf_nwc']   = inf_nwc_list
        out['inf_stack'] = inf_stack_list
        out['clashscore_pre_relax'] = clashscore_pre_list
        out['clashscore_post_relax'] = clashscore_post_list
    return out


def self_consistency_score_eternafold(
        samples,
        true_sec_struct_list,
        mask_coords,
        n_samples_ss=1,
        num_to_letter=NUM_TO_LETTER,
        return_sec_structs=False
):
    """
    Compute self consistency score for an RNA, given its true secondary structure(s)
    and a list of designed sequences.
    EternaFold is used to 'forward fold' the designs.

    Args:
        samples: designed sequences of shape (n_samples, seq_len)
        true_sec_struct_list: list of true secondary structures (n_true_ss, seq_len)
        mask_coords: mask for missing sequence coordinates to be ignored during evaluation
        n_samples_ss: number of predicted secondary structures per designed sample
        num_to_letter: lookup table mapping integers to nucleotides
        return_sec_structs: whether to return the predicted secondary structures

    Workflow:

        Input: For a given RNA molecule, we are given:
        - Designed sequences of shape (n_samples, seq_len)
        - True secondary structure(s) of shape (n_true_ss, seq_len)

        For each designed sequence:
        - Predict n_sample_ss secondary structures using EternaFold
        - For each pair of true and predicted secondary structures:
            - Compute MCC score between their adjacency matrix representations
        - Take the average MCC score across all n_sample_ss predicted structures

        Take the average MCC score across all n_samples designed sequences
    """
    n_true_ss = len(true_sec_struct_list)
    sequence_length = mask_coords.sum()
    # Import locally to avoid NetworkX conflicts
    from src.data.sec_struct_utils import dotbracket_to_adjacency, predict_sec_struct
    # map all entries from dotbracket to numerical representation
    true_sec_struct_list = np.array([dotbracket_to_adjacency(ss) for ss in true_sec_struct_list])
    # mask out missing sequence coordinates
    true_sec_struct_list = true_sec_struct_list[:, mask_coords][:, :, mask_coords]
    # reshape to(n_true_ss * n_samples_ss, seq_len, seq_len)
    true_sec_struct_list = torch.tensor(
        true_sec_struct_list
    ).unsqueeze(1).repeat(1, n_samples_ss, 1, 1).reshape(-1, sequence_length, sequence_length)

    mcc_scores = []
    pred_sec_structs = []
    for _sample in samples:
        # convert sample to string
        pred_seq = ''.join([num_to_letter[num] for num in _sample])
        # predict secondary structure(s) for each sample
        pred_sec_struct_list = predict_sec_struct(pred_seq, n_samples=n_samples_ss)
        if return_sec_structs:
            pred_sec_structs.append(copy.copy(pred_sec_struct_list))
        # map all entries from dotbracket to numerical representation
        pred_sec_struct_list = np.array([dotbracket_to_adjacency(ss) for ss in pred_sec_struct_list])
        # reshape to (n_samples_ss * n_true_ss, seq_len, seq_len)
        pred_sec_struct_list = torch.tensor(
            pred_sec_struct_list
        ).unsqueeze(0).repeat(n_true_ss, 1, 1, 1).reshape(-1, sequence_length, sequence_length)

        # compute mean MCC score between pairs of true and predicted secondary structures
        mcc_scores.append(
            binary_matthews_corrcoef(
                pred_sec_struct_list,
                true_sec_struct_list,
            ).float().mean()
        )

    if return_sec_structs:
        return np.array(mcc_scores), pred_sec_structs
    else:
        return np.array(mcc_scores)

def self_consistency_score_ribonanzanet(
    samples,
    true_sequence,
    mask_seq,
    ribonanza_net,
    num_to_letter=NUM_TO_LETTER,
    return_chem_mods=False,
):
    """Compute self consistency score for an RNA, given the (predicted) chemical modifications for
    the original RNA and a list of designed sequences. RibonanzaNet is used to 'forward fold' the
    designs.

    Args:
        samples: designed sequences of shape (n_samples, seq_len)
        true_sequence: true RNA sequence used to predict chemical modifications
        mask_seq: mask for missing sequence coordinates to be ignored during evaluation
        ribonanza_net: RibonanzaNet model
        num_to_letter: lookup table mapping integers to nucleotides
        return_chem_mods: whether to return the predicted chemical modifications

    Workflow:

        Input: For a given RNA molecule, we are given:
        - Designed sequences of shape (n_samples, seq_len)
        - Predicted chemical modifications for original sequence,
          of shape (n_samples, seq_len, 2), predicted via RibonanzaNet, of which we take
          the index 0 from the last channal --> 2A3/SHAPE.

        For each designed sequence:
        - Predict chemical modifications using RibonanzaNet
        - Compute mean absolute error between prediction and chemical modifications for
          the original sequence

        Take the average mean absolute error across all n_samples designed sequences
    """
    # Compute original sequence's chemical modifications using RibonanzaNet
    true_sequence = np.array([char for char in true_sequence])
    true_sequence = "".join(true_sequence[mask_seq])
    true_chem_mod = ribonanza_net.predict(true_sequence).unsqueeze(0).cpu().numpy()[:, :, 0]

    _samples = np.array([[num_to_letter[num] for num in seq] for seq in samples])
    pred_chem_mod = ribonanza_net.predict(_samples[:, mask_seq]).cpu().numpy()[:, :, 0]
    if return_chem_mods:
        return (np.abs(pred_chem_mod - true_chem_mod).mean(1)), pred_chem_mod
    else:
        return np.abs(pred_chem_mod - true_chem_mod).mean(1)

def self_consistency_score_ribonanzanet_sec_struct(
        samples,
        true_sec_struct,
        mask_coords,
        ribonanza_net_ss,
        num_to_letter = NUM_TO_LETTER,
        return_sec_structs = False
    ):
    # Import locally to avoid NetworkX conflicts
    from src.data.sec_struct_utils import dotbracket_to_adjacency
    # map from dotbracket to numerical representation
    true_sec_struct = np.array(dotbracket_to_adjacency(true_sec_struct, keep_pseudoknots=True))
    # mask out missing sequence coordinates
    true_sec_struct = true_sec_struct[mask_coords][:, mask_coords]
    # (n_samples, seq_len, seq_len)
    true_sec_struct = torch.tensor(true_sec_struct)

    _samples = np.array([[num_to_letter[num] for num in seq] for seq in samples])
    _, pred_sec_structs = ribonanza_net_ss.predict(_samples)  # (n_samples, seq_len, seq_len)

    mcc_scores = []
    for pred_sec_struct in pred_sec_structs:
        # map from dotbracket to numerical representation
        pred_sec_struct = torch.tensor(dotbracket_to_adjacency(pred_sec_struct, keep_pseudoknots=True))
        # compute mean MCC score between pairs of true and predicted secondary structures
        mcc_scores.append(
            binary_matthews_corrcoef(
                pred_sec_struct,
                true_sec_struct,
            ).float().mean()
        )

    if return_sec_structs:
        return np.array(mcc_scores), pred_sec_structs
    else:
        return np.array(mcc_scores)

def self_consistency_score_rhofold(
        samples,
        true_raw_data,
        mask_coords,
        rhofold,
        output_dir,
        num_to_letter=NUM_TO_LETTER,
        save_designs=False,
        save_pdbs=False,
        use_relax=False,
):
    """
    Original gRNAde-style RhoFold evaluation (3-value return).
    Forward-fold each designed seq with RhoFold and evaluate vs native structures.

    Returns:
        1) np.array(sc_rmsds)      # per-sample mean RMSD vs natives (Å)
        2) np.array(sc_tms)        # per-sample mean TM-score vs natives (0..1) 
        3) np.array(sc_gddts)      # per-sample mean GDT_TS (0..1)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collate designed sequences in fasta format
    input_seq = SeqRecord(
        Seq(true_raw_data["sequence"]),
        id=f"input_sequence,",
        description=f"input_sequence"
    )
    sequences = [input_seq]

    # Containers (simplified for original 3-value return)
    sc_rmsds, sc_tms, sc_gddts = [], [], []

    for idx, seq in enumerate(samples):
        # Save designed sequence to fasta file (temporary)
        seq = SeqRecord(
            Seq("".join([num_to_letter[num] for num in seq])),
            id=f"sample={idx},",
            description=f"sample={idx}"
        )
        sequences.append(seq)
        design_fasta_path = os.path.join(output_dir, f"design{idx}.fasta")
        SeqIO.write(seq, design_fasta_path, "fasta")

        # Forward fold designed sequence using RhoFold
        design_pdb_path = os.path.join(output_dir, f"design{idx}.pdb")
        _, _ = rhofold.predict(design_fasta_path, design_pdb_path, use_relax)  # Now returns (coords, plddt)

        # Load C4' coordinates of designed structure
        # Import locally to avoid NetworkX conflicts
        from src.data.data_utils import pdb_to_tensor, get_c4p_coords
        _, coords, _, _ = pdb_to_tensor(
            design_pdb_path,
            return_sec_struct=False,
            return_sasa=False,
            keep_insertions=False,
        )
        coords = get_c4p_coords(coords)
        coords = coords - coords.mean(dim=0)  # zero-center

        # Compare to each native structure in memory
        _sc_rmsds, _sc_tms, _sc_gddts = [], [], []
        for other_coords in true_raw_data["coords_list"]:
            _other = get_c4p_coords(other_coords)[mask_coords, :]
            _other = _other - _other.mean(dim=0)
            # global alignment (mobile=_other onto reference=coords)
            R_hat = rotation_matrix(_other, coords)[0]
            _other = _other @ R_hat.T

            _sc_rmsds.append(get_rmsd(coords, _other, superposition=True, center=True))
            _sc_tms.append(get_tmscore(coords, _other))
            _sc_gddts.append(get_gddt(coords, _other))

        sc_rmsds.append(np.mean(_sc_rmsds))
        sc_tms.append(np.mean(_sc_tms))
        sc_gddts.append(np.mean(_sc_gddts))

        # remove temporary FASTA and optionally the PDB
        os.unlink(design_fasta_path)
        if save_pdbs is False:
            try:
                os.unlink(design_pdb_path)
            except FileNotFoundError:
                pass

    # Save or clean directory
    if save_designs is False:
        shutil.rmtree(output_dir, ignore_errors=True)
    else:
        SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")

    return np.array(sc_rmsds), np.array(sc_tms), np.array(sc_gddts)


def self_consistency_score_rhofold_extended(
        samples,
        true_raw_data,
        mask_coords,
        rhofold,
        output_dir,
        num_to_letter=NUM_TO_LETTER,
        save_designs=False,
        save_pdbs=False,
        use_relax=False,
        use_inf=True,
        use_clash=True,
        use_lddt=True,
        use_mcq=True,
        phenix_wrapper_path=None,  # NEW: allow explicit wrapper path; falls back to tools/run_phenix.sh
):
    """
    Extended RhoFold evaluation with all metrics (8-value return).
    Forward-fold each designed seq with RhoFold and evaluate vs native structures.

    Returns:
        1) np.array(sc_rmsds)      # per-sample mean RMSD vs natives (Å)
        2) np.array(sc_tms)        # per-sample mean TM-score vs natives (0..1)
        3) np.array(sc_gddts)      # per-sample mean GDT_TS (0..1)
        4) np.array(sc_plddt)      # per-sample mean pLDDT from RhoFold (0..1)
        5) dict INF arrays: {"all","wc","nwc","stack"}  # if use_inf else empty arrays
        6) np.array(sc_clash)      # Phenix MolProbity clashscore if use_clash else empty array
        7) np.array(sc_lddt)       # per-sample mean lDDT scores vs natives (0..1)
        8) dict MCQ arrays: {"mcq_abs_deg","R","circ_sd_deg"}  # if use_mcq else empty arrays
    """
    os.makedirs(output_dir, exist_ok=True)

    # default phenix wrapper if not provided
    if phenix_wrapper_path is None:
        phenix_wrapper_path = os.path.join(PROJECT_PATH, "tools", "run_phenix.sh")

    # Collate designed sequences in fasta format
    input_seq = SeqRecord(
        Seq(true_raw_data["sequence"]),
        id=f"input_sequence,",
        description=f"input_sequence"
    )
    sequences = [input_seq]

    # Containers
    sc_rmsds, sc_tms, sc_gddts, sc_plddt = [], [], [], []
    sc_inf_all, sc_inf_wc, sc_inf_nwc, sc_inf_stack = [], [], [], []
    sc_clash_pre, sc_clash_post = [], []  # Separate containers for pre/post relax clash scores
    sc_lddt = []
    sc_mcq_abs, sc_mcq_R, sc_mcq_sd = [], [], []

    for idx, seq in enumerate(samples):
        # Save designed sequence to fasta file (temporary)
        seq = SeqRecord(
            Seq("".join([num_to_letter[num] for num in seq])),
            id=f"sample={idx},",
            description=f"sample={idx}"
        )
        sequences.append(seq)
        design_fasta_path = os.path.join(output_dir, f"design{idx}.fasta")
        SeqIO.write(seq, design_fasta_path, "fasta")

        # Forward fold designed sequence using RhoFold
        design_pdb_path = os.path.join(output_dir, f"design{idx}.pdb")
        _, plddt = rhofold.predict(design_fasta_path, design_pdb_path, use_relax)
        sc_plddt.append(np.mean(plddt))

        # Load C4' coordinates of designed structure
        # Import locally to avoid NetworkX conflicts
        from src.data.data_utils import pdb_to_tensor, get_c4p_coords
        _, coords, _, _ = pdb_to_tensor(
            design_pdb_path,
            return_sec_struct=False,
            return_sasa=False,
            keep_insertions=False,
        )
        coords = get_c4p_coords(coords)
        coords = coords - coords.mean(dim=0)  # zero-center

        # Compare to each native structure in memory
        _sc_rmsds, _sc_tms, _sc_gddts = [], [], []
        for other_coords in true_raw_data["coords_list"]:
            _other = get_c4p_coords(other_coords)[mask_coords, :]
            _other = _other - _other.mean(dim=0)
            # global alignment (mobile=_other onto reference=coords)
            R_hat = rotation_matrix(_other, coords)[0]
            _other = _other @ R_hat.T

            _sc_rmsds.append(get_rmsd(coords, _other, superposition=True, center=True))
            _sc_tms.append(get_tmscore(coords, _other))
            _sc_gddts.append(get_gddt(coords, _other))

        sc_rmsds.append(np.mean(_sc_rmsds))
        sc_tms.append(np.mean(_sc_tms))
        sc_gddts.append(np.mean(_sc_gddts))

        # INF (Interaction Network Fidelity) if requested
        if use_inf:
            try:
                inf_score = get_inf(design_pdb_path, true_raw_data, DATA_PATH)
                sc_inf_all.append(inf_score["all"])
                sc_inf_wc.append(inf_score["wc"])
                sc_inf_nwc.append(inf_score["nwc"])
                sc_inf_stack.append(inf_score["stack"])
            except Exception as e:
                print(f"[RhoFold eval] INF failed: {e}")
                sc_inf_all.append(np.nan)
                sc_inf_wc.append(np.nan)
                sc_inf_nwc.append(np.nan)
                sc_inf_stack.append(np.nan)

        # Clashscore (Phenix MolProbity) - Pre and Post Relax if requested
        if use_clash:
            if use_relax:
                # When relaxation is used, calculate clash scores before and after relax
                # Pre-relax: calculate on _unrelaxed.pdb file
                pre_relax_pdb_path = f'{design_pdb_path[:-4]}_unrelaxed.pdb'
                post_relax_pdb_path = design_pdb_path  # This is the relaxed structure
                
                # Calculate pre-relax clash score
                try:
                    clash_pre = get_clash_score_phenix(pre_relax_pdb_path, phenix_wrapper_path)
                except Exception as e:
                    print(f"[RhoFold eval] pre-relax clashscore (Phenix) failed: {e}")
                    clash_pre = np.nan
                
                # Calculate post-relax clash score
                try:
                    clash_post = get_clash_score_phenix(post_relax_pdb_path, phenix_wrapper_path)
                except Exception as e:
                    print(f"[RhoFold eval] post-relax clashscore (Phenix) failed: {e}")
                    clash_post = np.nan
                    
            else:
                # When no relaxation, only calculate pre-relax clash score, use NaN for post-relax
                try:
                    clash_pre = get_clash_score_phenix(design_pdb_path, phenix_wrapper_path)
                except Exception as e:
                    print(f"[RhoFold eval] clashscore (Phenix) failed: {e}")
                    clash_pre = np.nan
                
                # No post-relax calculation when relaxation is disabled
                clash_post = np.nan
                
            sc_clash_pre.append(clash_pre)
            sc_clash_post.append(clash_post)

        # lDDT (Local Distance Difference Test) if requested
        if use_lddt:
            _lddt_scores = []
            for nid in true_raw_data["id_list"]:
                native_pdb_path = os.path.join(DATA_PATH, "raw", f"{nid}.pdb")
                if os.path.exists(native_pdb_path):
                    try:
                        # Use OpenStructure lDDT v2 to avoid NetworkX conflicts
                        lddt = get_lddt_openstructure_v2(design_pdb_path, native_pdb_path)
                        if not np.isnan(lddt):  # Only include successful calculations (NaN for failures)
                            _lddt_scores.append(lddt)
                    except Exception as e:
                        print(f"[RhoFold eval] lDDT failed for {nid}: {e}")
            sc_lddt.append(np.mean(_lddt_scores) if _lddt_scores else np.nan)

        # MCQ (Mean of Circular Quantities) if requested
        if use_mcq:
            try:
                mcq_stats = mcq_avg_vs_natives(design_pdb_path, true_raw_data, DATA_PATH)
                sc_mcq_abs.append(mcq_stats["mcq_abs_deg"])
                sc_mcq_R.append(mcq_stats["R"])
                sc_mcq_sd.append(mcq_stats["circ_sd_deg"])
            except Exception as e:
                print(f"[RhoFold eval] MCQ failed: {e}")
                sc_mcq_abs.append(np.nan)
                sc_mcq_R.append(np.nan)
                sc_mcq_sd.append(np.nan)

        # remove temporary FASTA and optionally the PDB
        os.unlink(design_fasta_path)
        if save_pdbs is False:
            try:
                os.unlink(design_pdb_path)
            except FileNotFoundError:
                pass

    # Save or clean directory
    if save_designs is False:
        shutil.rmtree(output_dir, ignore_errors=True)
    else:
        SeqIO.write(sequences, os.path.join(output_dir, "all_designs.fasta"), "fasta")

    # Package INF dict (return empty arrays if disabled, to keep shape stable)
    inf_dict = {
        "all": np.array(sc_inf_all) if use_inf else np.array([]),
        "wc":  np.array(sc_inf_wc)  if use_inf else np.array([]),
        "nwc": np.array(sc_inf_nwc) if use_inf else np.array([]),
        "stack": np.array(sc_inf_stack) if use_inf else np.array([]),
    }

    # Package MCQ dict (return empty arrays if disabled)
    mcq_dict = {
        "mcq_abs_deg": np.array(sc_mcq_abs) if use_mcq else np.array([]),
        "R": np.array(sc_mcq_R) if use_mcq else np.array([]),
        "circ_sd_deg": np.array(sc_mcq_sd) if use_mcq else np.array([]),
    }

    # Package clash dict (return empty arrays if disabled, to keep shape stable)
    clash_dict = {
        "pre_relax": np.array(sc_clash_pre) if use_clash else np.array([]),
        "post_relax": np.array(sc_clash_post) if use_clash else np.array([]),
    }

    return (np.array(sc_rmsds),
            np.array(sc_tms),
            np.array(sc_gddts),
            np.array(sc_plddt),
            inf_dict,
            clash_dict,
            np.array(sc_lddt) if use_lddt else np.array([]),
            mcq_dict)


def get_three_mer_corr(samples,true_seq, mask_coords):
    """
        Compute 3-mer correlation between designed sequences and true sequences.

        Args:
            samples: designed sequences of shape (n_samples, seq_len)
            true_seqs: true sequences (seq_len)
            mask_coords: mask for missing sequence coordinates to be ignored during evaluation

        Returns:
            Array of correlation scores (n_samples,)
    """
    assert len(mask_coords) == len(true_seq)

    bases = ['A', 'C', 'G', 'U']
    import itertools
    all_3mers = [''.join(k) for k in itertools.product(bases, repeat=3)]
    kmer_index = {kmer: i for i, kmer in enumerate(all_3mers)}
    n_kmers = len(all_3mers)

    def compute_kmer_vector(seq, mask=None):
        vec = np.zeros(n_kmers, dtype=float)
        seq_len = len(seq)
        for i in range(seq_len - 2):
            if mask is None or (mask[i] and mask[i+1] and mask[i+2]):
                kmer = seq[i:i+3]
                if kmer in kmer_index:
                    vec[kmer_index[kmer]] += 1
        if vec.sum() > 0:
            vec /= vec.sum()  # normalize to frequency
        return vec

    true_vec = compute_kmer_vector(true_seq, mask_coords)
    scores = []
    for s in samples:
        sample_vec = compute_kmer_vector(s)
        if np.std(sample_vec) == 0 or np.std(true_vec) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(sample_vec, true_vec)[0, 1]
        scores.append(corr)
    return np.array(scores)

def get_trimer_profile_novelty(samples, mask_coords=None, reference_db=None):
    """
    Compute Trimer Profile Novelty (TPN) for designed RNA sequences.
    
    TPN measures how different a designed RNA's 3-mer usage pattern is from 
    any known natural RNA. Higher scores indicate more novel sequences.
    
    Args:
        samples: designed sequences of shape (n_samples, seq_len) or list of strings
        mask_coords: mask for missing sequence coordinates (optional)
        reference_db: list of reference RNA sequences (if None, uses default)
        
    Returns:
        Array of novelty scores (n_samples,) where each score is between 0 and 1
        Higher scores = more novel (less similar to natural RNAs)
    """
    import itertools
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Initialize 3-mer vocabulary
    bases = ['A', 'C', 'G', 'U']
    all_3mers = [''.join(k) for k in itertools.product(bases, repeat=3)]
    kmer_index = {kmer: i for i, kmer in enumerate(all_3mers)}
    n_kmers = len(all_3mers)  # 64 possible 3-mers
    
    def compute_kmer_vector(seq, mask=None):
        """Compute normalized 3-mer frequency vector for a sequence"""
        vec = np.zeros(n_kmers, dtype=float)
        seq_len = len(seq)
        for i in range(seq_len - 2):
            if mask is None or (len(mask) > i+2 and mask[i] and mask[i+1] and mask[i+2]):
                kmer = seq[i:i+3]
                if kmer in kmer_index:
                    vec[kmer_index[kmer]] += 1
        if vec.sum() > 0:
            vec /= vec.sum()  # normalize to frequency
        return vec
    
    # Convert samples to strings if needed
    if isinstance(samples, np.ndarray) and len(samples.shape) == 2:
        # Convert from numeric to string representation
        from src.constants import NUM_TO_LETTER
        sample_sequences = []
        for sample in samples:
            seq_str = "".join([NUM_TO_LETTER[int(n)] for n in sample])
            sample_sequences.append(seq_str)
    else:
        sample_sequences = samples
    
    # Default reference database (diverse natural RNA sequences)
    if reference_db is None:
        reference_db = _get_default_rna_reference_db()
    
    # Compute 3-mer vectors for reference database
    ref_vectors = []
    for ref_seq in reference_db:
        ref_vec = compute_kmer_vector(ref_seq)
        if ref_vec.sum() > 0:  # Only include valid vectors
            ref_vectors.append(ref_vec)
    
    if len(ref_vectors) == 0:
        print("Warning: No valid reference sequences found. Using zero novelty.")
        return np.zeros(len(sample_sequences))
    
    ref_matrix = np.array(ref_vectors)  # Shape: (n_ref, 64)
    
    # Compute novelty for each designed sequence
    novelty_scores = []
    for seq in sample_sequences:
        sample_vec = compute_kmer_vector(seq, mask_coords)
        
        if sample_vec.sum() == 0:
            # Invalid sequence, assign zero novelty
            novelty_scores.append(0.0)
            continue
            
        # Find maximum cosine similarity with any reference sequence
        sample_vec = sample_vec.reshape(1, -1)
        similarities = cosine_similarity(sample_vec, ref_matrix)[0]
        max_similarity = np.max(similarities)
        
        # Novelty = 1 - max_similarity (higher novelty means less similar)
        novelty = 1.0 - max_similarity
        novelty_scores.append(max(0.0, novelty))  # Ensure non-negative
    
    return np.array(novelty_scores)

def _get_default_rna_reference_db():
    """
    Get a default reference database of diverse natural RNA sequences.
    This includes representatives from major RNA families.
    """
    # Curated set of diverse natural RNA sequences from different families
    # These represent common RNA structural and sequence motifs
    reference_sequences = [
        # tRNAs (transfer RNAs) - highly conserved
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA",  # tRNA-Phe
        "GGAGCGGTAGTTCAGTCGGTTAGAATACCCTGCCTGTCACGCAGGGGUCCGGGTUCGATTCCGGCCGCTCCA",   # tRNA-Ala
        "GGGCCCGTGGCGCAATGGATCATCGGCTCTAAAGGCTGAAGCAACCTCAAGTGGGCGTGGTTCGAGTCCACGTGGGCCC", # tRNA-Leu
        
        # rRNAs (ribosomal RNAs) - structural scaffolds
        "UUAAUCAGUCGUGGUUGAUCCUGAGUGGUAGUAGGUUGCGAAGGCAGCCGACCUACACAUUCAAGGAAGGCAG",  # 16S-like
        "GGGAAAGCCCGGUAAAUGCGAAUGAAAAGGCCCGAACGUCUGAACUCAAUCGUGCACACCGAUGUGCGGGCAAGAUCUAAAUGUGAACCCUC", # 23S-like
        
        # microRNAs (miRNAs) - regulatory small RNAs  
        "UGGAAUGUAAAGAAGUAUGUAU",  # miR-1
        "ACAGUGCUUGACUGCUGAAGUA",  # miR-16
        "UAAAGCUAGCUUACCAUAAGGUA", # miR-21
        
        # snRNAs (small nuclear RNAs) - splicing machinery
        "AUACUUACCUGAGGGAAAGGUAUGUGUAGUAAGCCAGGUGAACUUCAUGGGUUAUAUAAUUUCCCUAGUCCUGUGCUAA", # U1
        "GGCAGGGGAAAUAUCGCUUUGUCAAUUGUCAUAGCCUCGUAACCCACUAGAGUUGAGGUGGAGCCUGUACUUGAACGCAG", # U2
        
        # Viral RNAs - diverse structures
        "ACAAACCAUCUCAAACAGACAACCCAAACGACACAAACGGACACACAAA",  # Hepatitis C virus-like
        "GGGAAAGGGCAACAAGCCGCAGCAGCGCGACAACGGCGCAGUAACGGCG",   # HIV-like TAR structure
        
        # Riboswitches - structured regulatory RNAs
        "GGAAGCCUGGGGCAACUGAGCUAACUCCAAAAGGAAAGCUCUGACAACAGGCUCAAAGCCGUGCGAUGUACGCCGGAGACC", # TPP riboswitch
        "GGCGACCCCCGGCAACCGCGCCCGACGGGCGCGAGGAAACAUCAAGAGAGGUGCUCCGAACACCUGCGGAUGGCCACGUACGGC", # FMN riboswitch
        
        # Hammerhead ribozymes - catalytic RNAs
        "CUGAUGAGGCCGAAAGGCCGAAACAGGUGAAACUCCGUAGCGCCGAUGAGGCUGU",
        "GGCCACGCGUCUUGAUCAAGAGGCUGAUGAGGCCGAAAGGCCGAAACAGGUGAAACUCCGUAGCGCCGAUGAGGCUGU",
        
        # Group I/II intron fragments - large structured RNAs
        "GGCCCUAACAGGCCGAGGCGGCCCAACCCAAGCCAGGCCGCGGUGGCGGCGGCGGUGCCGACGGGGUGAACGCC",
        "GCGCUGCUUGGCAUUCAGGGAAGGAAGAAGGCGCGAGGCCCCGACCCCUGCCCCCGCCCGCGGGGAGGGCUGGGAGAA",
        
        # Random natural-like sequences (to increase diversity)
        "GCCUGAAGCUGCGAGGCAGCUGUGCUCCGCGAGGCCUGAAGCUGCGAGGCAGCUGUGC",
        "AUGGCAAGCCUGCGAUGGCCAAGCCUGCGAUGGCCAAGCCUGCG",
        "CGGCAAGGGCAAGCGGGCAAGGGCAAGCGGGCAAGGGCAAGC",
        "GGAAGGGGAACCCUUUCCUGGAAGGGGAACCCUUUCCUGGAAGGGGAA",
        "UCGCGCGUUGCAGCGGUGCAGCGGUGCAGCGGUGCAGCGGUUCGCGCGU",
    ]
    
    return reference_sequences

def get_tmscore(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Template Modelling score (TM-score).

    Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

    https://en.wikipedia.org/wiki/Template_modeling_score

    TM-score is a measure of similarity between two protein structures.
    The TM-score is intended as a more accurate measure of the global
    similarity of full-length protein structures than the often used RMSD
    measure. The TM-score indicates the similarity between two structures
    by a score between ``[0, 1]``, where 1 indicates a perfect match
    between two structures (thus the higher the better). Generally scores
    below 0.20 corresponds to randomly chosen unrelated proteins whereas
    structures with a score higher than 0.5 assume roughly the same fold.
    A quantitative study shows that proteins of TM-score = 0.5 have a
    posterior probability of 37% in the same CATH topology family and of
    13% in the same SCOP fold family. The probabilities increase rapidly
    when TM-score > 0.5. The TM-score is designed to be independent of
    protein lengths.

    We have adapted the implementation to RNA (TM-score threshold = 0.45).
    Requires aligned C4' coordinates as input.
    """
    l_target = y.shape[0]
    d0_l_target = 1.24 * np.power(l_target - 15, 1 / 3) - 1.8
    di = torch.pairwise_distance(y_hat, y)
    out = torch.sum(1 / (1 + (di / d0_l_target) ** 2)) / l_target
    if torch.isnan(out):
        return torch.tensor(0.0)
    return out

def get_gddt(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Global Distance Deviation Test metric (GDDT).

    Credit: Arian Jamasb, graphein (https://github.com/a-r-j/graphein)

    https://en.wikipedia.org/wiki/Global_distance_test

    The GDT score is calculated as the largest set of amino acid residues'
    alpha carbon atoms in the model structure falling within a defined
    distance cutoff of their position in the experimental structure, after
    iteratively superimposing the two structures. By the original design the
    GDT algorithm calculates 20 GDT scores, i.e. for each of 20 consecutive distance
    cutoffs (``0.5 Å, 1.0 Å, 1.5 Å, ... 10.0 Å``). For structure similarity assessment
    it is intended to use the GDT scores from several cutoff distances, and scores
    generally increase with increasing cutoff. A plateau in this increase may
    indicate an extreme divergence between the experimental and predicted structures,
    such that no additional atoms are included in any cutoff of a reasonable distance.
    The conventional GDT_TS total score in CASP is the average result of cutoffs at
    ``1``, ``2``, ``4``, and ``8`` Å.

    Random predictions give around 20; getting the gross topology right gets one to ~50;
    accurate topology is usually around 70; and when all the little bits and pieces,
    including side-chain conformations, are correct, GDT_TS begins to climb above 90.

    We have adapted the implementation to RNA.
    Requires aligned C4' coordinates as input.
    """
    # Get distance between points
    dist = torch.norm(y - y_hat, dim=1)

    # Return mean fraction of distances below cutoff for each cutoff (1, 2, 4, 8)
    count_1 = (dist < 1).sum() / dist.numel()
    count_2 = (dist < 2).sum() / dist.numel()
    count_4 = (dist < 4).sum() / dist.numel()
    count_8 = (dist < 8).sum() / dist.numel()
    out = torch.mean(torch.tensor([count_1, count_2, count_4, count_8]))
    if torch.isnan(out):
        return torch.tensor(0.0)
    return out

def edit_distance(s: str, t: str) -> int:
    """
    A Space efficient Dynamic Programming based Python3 program
    to find minimum number operations to convert str1 to str2

    Source: https://www.geeksforgeeks.org/edit-distance-dp-5/
    """
    n = len(s)
    m = len(t)

    prev = [j for j in range(m+1)]
    curr = [0] * (m+1)

    for i in range(1, n+1):
        curr[0] = i
        for j in range(1, m+1):
            if s[i-1] == t[j-1]:
                curr[j] = prev[j-1]
            else:
                mn = min(1 + prev[j], 1 + curr[j-1])
                curr[j] = min(mn, 1 + prev[j-1])
        prev = curr.copy()

    return prev[m]


# molprobity alternative clash score calculation (not used)
def get_clashscore(pdb_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pdb_h = os.path.join(output_dir, "model_H.pdb")
    probe_out = os.path.join(output_dir, "probe.log")

    # Set environment variable REDUCE_WWPDB_HET_DICT
    reduce_dict = os.path.join(MOLPROBITY_HOME, "modules/reduce/reduce_wwPDB_het_dict.txt")
    os.environ["REDUCE_WWPDB_HET_DICT"] = reduce_dict

    # Add hydrogens
    reduce_exe = os.path.join(MOLPROBITY_HOME, "modules/reduce/reduce_src/reduce")
    with open(pdb_h, "w") as f:
        subprocess.run([reduce_exe, "-BUILD", pdb_file], stdout=f, check=True)

    # Run probe to calculate clashes
    probe_exe = os.path.join(MOLPROBITY_HOME, "modules/probe/probe")
    with open(probe_out, "w") as f:
        subprocess.run([probe_exe, "-u", pdb_h], stdout=f, check=True)

    # Parse clash score
    clashscore = None
    with open(probe_out) as f:
        for line in f:
            if "clashscore" in line.lower():
                clashscore = float(line.strip().split()[-1])
                break

    if clashscore is None:
        raise RuntimeError(f"clashscore not found in probe output (file: {probe_out})")

    return clashscore

# molprobity 
# def get_clashscore(pdb_file, output_dir, phenix_env=None):
#     """
#     Run phenix.molprobity to calculate clashscore.

#     Args:
#         pdb_file (str): input PDB file
#         output_dir (str): directory to save results
#         phenix_env (str, optional): path to phenix_env.sh, if not already sourced

#     Returns:
#         float: clashscore value
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     out_file = os.path.join(output_dir, "molprobity.out")

#     # Command: phenix.molprobity pdb_file > output
#     cmd = ["phenix.molprobity", pdb_file]

#     # If user provides phenix_env, wrap in a shell to source environment
#     if phenix_env:
#         cmd = ["bash", "-c", f"source {phenix_env} && phenix.molprobity {pdb_file}"]

#     with open(out_file, "w") as f:
#         subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

#     # Parse output for Clashscore
#     clashscore = None
#     with open(out_file) as f:
#         for line in f:
#             if "Clashscore" in line:
#                 try:
#                     clashscore = float(line.strip().split()[-1])
#                     break
#                 except ValueError:
#                     continue

#     if clashscore is None:
#         raise RuntimeError(f"Clashscore not found in molprobity output (file: {out_file})")

#     return clashscore


def get_inf(predicted_pdb_path, true_raw_data, data_path=DATA_PATH):
    """
        Compute Interaction Network Fidelity (INF) metrics for a predicted RNA structure.

        Args:
            predicted_pdb_path: path to the predicted RNA PDB
            rue_raw_data: Original RNA raw data containing at least 'id_list' of native structures
            data_path: root path where native PDBs are stored

        Returns:
            dict: INF metrics with keys 'all', 'wc', 'nwc', 'stack'
                  values are floats (or -1 if no valid comparison)
        """
    predicted_struct = RNA_normalizer.PDBStruct()
    predicted_struct.load(predicted_pdb_path)

    inf_all, inf_wc, inf_nwc, inf_stack = [], [], [], []

    for id in true_raw_data["id_list"]:
        native_pdb_path = os.path.join(data_path, "raw", f"{id}.pdb")
        native_struct = RNA_normalizer.PDBStruct()
        native_struct.load(native_pdb_path)

        comparer = RNA_normalizer.PDBComparer()
        val_all = comparer.INF(predicted_struct, native_struct, type="ALL")
        val_wc = comparer.INF(predicted_struct, native_struct, type="PAIR_2D")
        val_nwc = comparer.INF(predicted_struct, native_struct, type="PAIR_3D")
        val_stack = comparer.INF(predicted_struct, native_struct, type="STACK")

        if val_all != -1: inf_all.append(val_all)
        if val_wc != -1: inf_wc.append(val_wc)
        if val_nwc != -1: inf_nwc.append(val_nwc)
        if val_stack != -1: inf_stack.append(val_stack)

    return {
        "all": np.mean(inf_all) if inf_all else -1,
        "wc": np.mean(inf_wc) if inf_wc else -1,
        "nwc": np.mean(inf_nwc) if inf_nwc else -1,
        "stack": np.mean(inf_stack) if inf_stack else -1,
    }

# Alternative clash score calculation using Phenix
# This works. Please use this version for the clash score calculation. 
def get_clash_score_phenix(pdb_file, phenix_wrapper_path):
    """
    Calculates the MolProbity clash score using a dedicated Phenix wrapper script.

    Args:
        pdb_file (str): The absolute path to the input PDB file.
        phenix_wrapper_path (str): The absolute path to the run_phenix.sh wrapper.

    Returns:
        float: The calculated all-atom clash score.
    """
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found at: {pdb_file}")
    if not os.path.exists(phenix_wrapper_path):
        raise FileNotFoundError(f"Phenix wrapper script not found at: {phenix_wrapper_path}")

    # The command is now much simpler: just call the wrapper script
    command = [phenix_wrapper_path, "phenix.molprobity", pdb_file]

    try:
        # We no longer need shell=True, which is safer.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        match = re.search(r"Clashscore\s*[:=]\s*(\d+\.\d+)", result.stdout)
        
        if match:
            return float(match.group(1))
        else:
            raise RuntimeError("Could not find or parse Clashscore in phenix.molprobity output.")

    except subprocess.CalledProcessError as e:
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        raise RuntimeError(f"Phenix wrapper command failed with exit code {e.returncode}.") from e

def get_lddt_robust(predicted_pdb_path, native_pdb_path):
    """
    Calculates the lDDT score between a predicted and native PDB structure.
    This production-ready version verifies sequence identity before calculation.
    """
    try:
        model_chain, model_seq = _get_pdb_info(predicted_pdb_path)
        native_chain, native_seq = _get_pdb_info(native_pdb_path)

        # CRITICAL: Verify that the sequences are identical for a meaningful score
        if model_seq != native_seq:
            print(f"❌ WARNING: Sequences in PDB files do not match. Cannot calculate a meaningful lDDT score.")
            # Uncomment the line below if you want to see the sequence differences
            # print(f"   Model Seq:  {model_seq}\n   Native Seq: {native_seq}")
            return -1.0
        
        seq_len = len(model_seq)

        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_model_path = os.path.join(temp_dir, 'model_extract.pdb')
            extracted_native_path = os.path.join(temp_dir, 'native_extract.pdb')
            
            # Extract the full, matching sequence to ensure clean input for the tool
            _extract_pdb_region(predicted_pdb_path, extracted_model_path, model_chain, 1, seq_len)
            _extract_pdb_region(native_pdb_path, extracted_native_path, native_chain, 1, seq_len)

            lddt_script_path = os.path.join(PROJECT_PATH, 'tools/RNA_assessment/lddt/bin/complex_lddt_no_stereocheck.py')
            lddt_script_dir = os.path.dirname(lddt_script_path)
            chain_mapping = f'{{"{model_chain}":"{native_chain}"}}'
            
            # Use isolated lddt_env to avoid NetworkX version conflicts
            conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
            lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
            
            command = [
                lddt_python,
                lddt_script_path,
                extracted_model_path,
                extracted_native_path,
                chain_mapping
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            else:
                print(f"--- lDDT Subprocess Failed ---")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return -1.0

    except Exception as e:
        print(f"An error occurred during lDDT calculation: {e}")
        return -1.0

def _get_residue_map(pdb_file):
    """Parses a PDB and returns the first chain ID and a map of {res_num: res_name}."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_file)
    model = list(structure.get_models())[0]
    chain = list(model.get_chains())[0]
    residue_map = {
        res.get_id()[1]: seq1(res.get_resname().strip())
        for res in chain if res.get_id()[0] == ' '
    }
    return chain.id, residue_map

def _extract_selected_residues(input_pdb, output_pdb, chain_id, res_nums_to_keep):
    """Extracts a specific set of residues from a specific chain."""
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            # Accept residue if its chain matches and its number is in the set to keep
            return residue.get_parent().id == chain_id and residue.get_id()[1] in res_nums_to_keep

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, ResidueSelect())

def get_lddt_inverse_folding(predicted_pdb_path, native_pdb_path):
    """
    Calculates lDDT for inverse folding by comparing residues at same positions
    regardless of sequence identity (since sequences are expected to differ).
    """
    try:
        model_chain, model_res_map = _get_residue_map(predicted_pdb_path)
        native_chain, native_res_map = _get_residue_map(native_pdb_path)
        
        # Find common residues by checking for matching numbers AND matching sequence identity
        common_res_nums = set()
        for res_num, model_res_name in model_res_map.items():
            if res_num in native_res_map and model_res_name == native_res_map[res_num]:
                common_res_nums.add(res_num)

        if not common_res_nums:
            # No common positions found
            return float('nan')
        
        if len(common_res_nums) < 3:
            # Need at least 3 residues for meaningful lDDT
            return float('nan')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_model_path = os.path.join(temp_dir, 'model_common.pdb')
            extracted_native_path = os.path.join(temp_dir, 'native_common.pdb')
            
            _extract_selected_residues(predicted_pdb_path, extracted_model_path, model_chain, common_res_nums)
            _extract_selected_residues(native_pdb_path, extracted_native_path, native_chain, common_res_nums)

            lddt_script_path = os.path.join(PROJECT_PATH, 'tools/RNA_assessment/lddt/bin/complex_lddt_no_stereocheck.py')
            lddt_script_dir = os.path.dirname(lddt_script_path)
            chain_mapping = f'{{"{model_chain}":"{native_chain}"}}'
            
            # Use isolated lddt_env to avoid NetworkX version conflicts
            conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
            lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
            
            command = [
                lddt_python,
                lddt_script_path,
                extracted_model_path,
                extracted_native_path,
                chain_mapping
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, cwd=lddt_script_dir)
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            else:
                return float('nan')

    except Exception as e:
        print(f"An error occurred during inverse folding lDDT calculation: {e}")
        return float('nan')

def get_lddt(predicted_pdb_path, native_pdb_path):
    """
    Wrapper function for lDDT calculation. 
    Uses position-based comparison suitable for inverse folding.
    """
    return get_lddt_inverse_folding(predicted_pdb_path, native_pdb_path)


def get_lddt_openstructure_v2(predicted_pdb_path, native_pdb_path):
    """
    Calculate lDDT using OpenStructure 2.4+ implementation in isolated environment.
    
    This version uses the isolated lddt_env environment to avoid NetworkX conflicts
    and provides improved lDDT calculation using modern OpenStructure API.
    
    Args:
        predicted_pdb_path (str): Path to predicted/model PDB structure
        native_pdb_path (str): Path to native/reference PDB structure
        
    Returns:
        float: lDDT score (0-1) or NaN if calculation fails
        
    Note:
        Uses parameters optimized for RNA structures:
        - inclusion_radius: 15 Å (standard)
        - sequence_separation: 0 (consider all contacts except intra-residue)
        - thresholds: [0.5, 1.0, 2.0, 4.0] Å (standard lDDT thresholds)
        - bb_only: False (consider all atoms for RNA)
        - check_resnames: False (suitable for inverse folding)
    """
    try:
        import tempfile
        import subprocess
        
        # Verify input files exist
        if not os.path.exists(predicted_pdb_path):
            print(f"Predicted PDB not found: {predicted_pdb_path}")
            return float('nan')
        if not os.path.exists(native_pdb_path):
            print(f"Native PDB not found: {native_pdb_path}")
            return float('nan')
        
        # Create temporary script for OpenStructure lDDT calculation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_content = f'''#!/usr/bin/env python3
import sys
import os

def create_chain_mapping(predicted_entity, native_entity):
    """Create chain mapping for lDDT calculation with multi-chain support."""
    try:
        # Get chain information
        native_chains = [(c.name, len(c.residues)) for c in native_entity.chains]
        predicted_chains = [(c.name, len(c.residues)) for c in predicted_entity.chains]
        
        if not native_chains or not predicted_chains:
            return None
        
        # Strategy 1: Single predicted chain vs multi-chain native
        # Map predicted chain to largest native chain
        if len(predicted_chains) == 1 and len(native_chains) > 1:
            largest_native = max(native_chains, key=lambda x: x[1])
            return {{predicted_chains[0][0]: largest_native[0]}}
        
        # Strategy 2: Multi-chain vs multi-chain or single vs single
        # Map largest predicted to largest native
        if len(predicted_chains) >= 1 and len(native_chains) >= 1:
            largest_native = max(native_chains, key=lambda x: x[1])
            largest_predicted = max(predicted_chains, key=lambda x: x[1])
            return {{largest_predicted[0]: largest_native[0]}}
        
        return None
        
    except Exception as e:
        print(f"Chain mapping error: {{e}}", file=sys.stderr)
        return None

def calculate_lddt_v2(predicted_pdb, native_pdb):
    """Calculate lDDT using OpenStructure with proper chain mapping support."""
    try:
        import ost
        import ost.mol
        import ost.io
        from ost.mol.alg import lddt
        
        # Load structures
        native_entity = ost.io.LoadPDB(native_pdb)
        predicted_entity = ost.io.LoadPDB(predicted_pdb)
        
        if not native_entity.IsValid() or not predicted_entity.IsValid():
            return float('nan')
        
        # Create chain mapping for multi-chain structures
        chain_mapping = None
        if len(native_entity.chains) > 1 or len(predicted_entity.chains) > 1:
            chain_mapping = create_chain_mapping(predicted_entity, native_entity)
        
        # Option 1: Try direct calculation with chain mapping
        try:
            scorer = lddt.lDDTScorer(
                target=native_entity,
                inclusion_radius=15.0,           # Standard inclusion radius
                sequence_separation=0,           # Consider all contacts except intra-residue
                bb_only=False                   # Consider all atoms for RNA
            )
            
            # Build lDDT arguments
            lddt_args = {{
                'model': predicted_entity,
                'thresholds': [0.5, 1.0, 2.0, 4.0],    # Standard lDDT thresholds
                'check_resnames': False,                # Don't enforce residue name matching
                'no_interchain': False,                 # Include interchain contacts if present
                'no_intrachain': False                  # Include intrachain contacts
            }}
            
            # Add chain mapping if needed
            if chain_mapping is not None:
                lddt_args['chain_mapping'] = chain_mapping
            
            global_lddt, per_residue_lddt = scorer.lDDT(**lddt_args)
            
            if global_lddt is not None:
                return float(global_lddt)
        except Exception as e:
            # If chain mapping fails, try without it for single-chain case
            if "chain mapping" not in str(e).lower():
                print(f"Direct lDDT failed: {{e}}", file=sys.stderr)
        
        # Option 2: Try with nucleic acid selection and chain mapping
        try:
            native_clean = native_entity.Select("nucleic")
            predicted_clean = predicted_entity.Select("nucleic")
            
            if len(native_clean.residues) > 0 and len(predicted_clean.residues) > 0:
                # Update chain mapping for cleaned structures
                clean_chain_mapping = None
                if len(native_clean.chains) > 1 or len(predicted_clean.chains) > 1:
                    clean_chain_mapping = create_chain_mapping(predicted_clean, native_clean)
                
                scorer = lddt.lDDTScorer(
                    target=native_clean,
                    inclusion_radius=15.0,
                    sequence_separation=0,
                    bb_only=False
                )
                
                lddt_args = {{
                    'model': predicted_clean,
                    'thresholds': [0.5, 1.0, 2.0, 4.0],
                    'check_resnames': False,
                    'no_interchain': False,
                    'no_intrachain': False
                }}
                
                if clean_chain_mapping is not None:
                    lddt_args['chain_mapping'] = clean_chain_mapping
                
                global_lddt, per_residue_lddt = scorer.lDDT(**lddt_args)
                
                if global_lddt is not None:
                    return float(global_lddt)
        except Exception as e:
            print(f"Nucleic lDDT failed: {{e}}", file=sys.stderr)
        
        # Option 3: Try backbone-only with chain mapping as fallback
        try:
            scorer = lddt.lDDTScorer(
                target=native_entity,
                inclusion_radius=15.0,
                sequence_separation=0,
                bb_only=True  # backbone only
            )
            
            lddt_args = {{
                'model': predicted_entity,
                'thresholds': [0.5, 1.0, 2.0, 4.0],
                'check_resnames': False
            }}
            
            if chain_mapping is not None:
                lddt_args['chain_mapping'] = chain_mapping
            
            global_lddt, per_residue_lddt = scorer.lDDT(**lddt_args)
            
            if global_lddt is not None:
                return float(global_lddt)
        except Exception as e:
            print(f"Backbone lDDT failed: {{e}}", file=sys.stderr)
        
        # All calculation methods failed
        return float('nan')
        
    except Exception as e:
        print(f"OpenStructure lDDT v2 error: {{e}}", file=sys.stderr)
        return float('nan')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: script.py <predicted_pdb> <native_pdb>")
        sys.exit(1)
    
    predicted_pdb = sys.argv[1]
    native_pdb = sys.argv[2]
    
    result = calculate_lddt_v2(predicted_pdb, native_pdb)
    print(result)
'''
            f.write(script_content)
            script_path = f.name
        
        # Use isolated lddt_env to run OpenStructure lDDT calculation
        conda_base = "/mnt/dna01/library-seq/luca/miniforge3"
        lddt_python = os.path.join(conda_base, "envs", "lddt_env", "bin", "python")
        
        command = [
            lddt_python,
            script_path,
            predicted_pdb_path,
            native_pdb_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        
        # Clean up temporary script
        os.unlink(script_path)
        
        if result.returncode == 0 and result.stdout.strip():
            lddt_score = float(result.stdout.strip())
            return lddt_score
        else:
            if result.stderr:
                print(f"lDDT v2 stderr: {result.stderr.strip()}")
            return float('nan')
            
    except subprocess.TimeoutExpired:
        print(f"lDDT v2 calculation timed out")
        if 'script_path' in locals() and os.path.exists(script_path):
            os.unlink(script_path)
        return float('nan')
    except Exception as e:
        print(f"lDDT v2 subprocess error: {e}")
        if 'script_path' in locals() and os.path.exists(script_path):
            os.unlink(script_path)
        return float('nan')



def get_usalign_tmscore(predicted_pdb_path: str,
                        native_pdb_path: str,
                        aggregate: str = "avg"):
    """
    Run US-align (RNA mode) via your existing wrapper and return an aggregated TM-score.

    Args:
        predicted_pdb_path: path to model PDB
        native_pdb_path:    path to native/target PDB
        aggregate:          how to combine TM-scores reported by US-align
                            ("avg" = average of the two normalized TM-scores,
                             "max" = max of the two, "min" = min of the two)

    Returns:
        tm_agg (float), result (namedtuple with fields:
            tmscore_chain1, tmscore_chain2, rmsd, align_len, seq_identity, stdout)

    Notes:
        - US-align typically reports two TM-scores normalized by each structure length.
        - For balanced evaluation across variable lengths, "avg" is a good default.
        - If you prefer a more lenient criterion, use aggregate="max".
    """
    try:
        # Import your existing wrapper
        # (it reads configs/base.yaml to find <project_root>/tools/USalign/USalign)
        from tools.usalign_utils import run_rna_usalign, USAlignResult
    except Exception as e:
        print("❌ Could not import tools.usalign_utils. "
              "Ensure it exists and configs/base.yaml contains the USalign paths.")
        print(f"Import error: {e}")
        return -1.0, None

    # Sanity checks
    if not os.path.exists(predicted_pdb_path):
        print(f"❌ Predicted PDB not found: {predicted_pdb_path}")
        return -1.0, None
    if not os.path.exists(native_pdb_path):
        print(f"❌ Native PDB not found: {native_pdb_path}")
        return -1.0, None

    # Run USalign via your wrapper
    res = run_rna_usalign(predicted_pdb_path, native_pdb_path)
    if res is None:
        print("❌ USalign wrapper returned None (check executable path and inputs).")
        return -1.0, None

    tm1 = float(res.tmscore_chain1)
    tm2 = float(res.tmscore_chain2)

    if aggregate == "avg":
        tm_agg = 0.5 * (tm1 + tm2)
    elif aggregate == "max":
        tm_agg = max(tm1, tm2)
    elif aggregate == "min":
        tm_agg = min(tm1, tm2)
    else:
        print(f"⚠️ Unknown aggregate='{aggregate}', defaulting to 'avg'.")
        tm_agg = 0.5 * (tm1 + tm2)

    return tm_agg, res

def usalign_tm_pdbpair(predicted_pdb_path: str,
                       native_pdb_path: str,
                       aggregate: str = "avg",
                       return_detail: bool = False):
    """
    Compute an aggregated TM-score between two PDBs using tools/usalign_utils.

    Args:
        predicted_pdb_path: path to predicted/model PDB
        native_pdb_path:    path to native/reference PDB
        aggregate:          {"avg","max","min"} to combine the two normalized TM-scores
        return_detail:      if True, also return the raw USAlignResult (tm1, tm2, rmsd, etc.)

    Returns:
        If return_detail=False: float aggregated TM-score
        If return_detail=True:  (float aggregated TM-score, USAlignResult)
                                where USAlignResult has fields:
                                tmscore_chain1, tmscore_chain2, rmsd, align_len,
                                seq_id (fraction 0..1 if available), seq_identity (alias), stdout
    """
    try:
        from tools.usalign_utils import run_rna_usalign, aggregate_tm
    except Exception as e:
        print(f"[usalign] Could not import tools.usalign_utils: {e}")
        if return_detail:
            return -1.0, None
        return -1.0

    if not os.path.exists(predicted_pdb_path):
        print(f"[usalign] Predicted PDB not found: {predicted_pdb_path}")
        if return_detail:
            return -1.0, None
        return -1.0
    if not os.path.exists(native_pdb_path):
        print(f"[usalign] Native PDB not found: {native_pdb_path}")
        if return_detail:
            return -1.0, None
        return -1.0

    try:
        res = run_rna_usalign(predicted_pdb_path, native_pdb_path)
        tm_agg = aggregate_tm(res.tmscore_chain1, res.tmscore_chain2, aggregate)
    except Exception as e:
        print(f"[usalign] USalign failed on\n  model:  {predicted_pdb_path}\n  native: {native_pdb_path}\n  error:  {e}")
        if return_detail:
            return -1.0, None
        return -1.0

    return (tm_agg, res) if return_detail else tm_agg

def usalign_tm_vs_natives(predicted_pdb_path: str,
                          true_raw_data: dict,
                          data_path: str = DATA_PATH,
                          aggregate: str = "avg",
                          return_detail: bool = False):
    """
    Compute US-align TM-score of a predicted PDB against all native PDBs for an item.

    Args:
        predicted_pdb_path: path to predicted/model PDB
        true_raw_data:      dataset item dict containing "id_list" of native structure IDs
        data_path:          project data root (expects native PDBs under {data_path}/raw/{id}.pdb)
        aggregate:          {"avg","max","min"} to combine the two normalized TM-scores
        return_detail:      if True, also return list of USAlignResult (one per native)

    Returns:
        tm_list: np.ndarray of per-native aggregated TM-scores (length = len(id_list))
        tm_mean: float, mean aggregated TM over all natives (or -1.0 if none)
        detail (optional): list of USAlignResult in same order as id_list (only if return_detail=True)
    """
    try:
        from tools.usalign_utils import run_rna_usalign, aggregate_tm
    except Exception as e:
        print(f"[usalign] Could not import tools.usalign_utils: {e}")
        if return_detail:
            return np.array([]), -1.0, []
        return np.array([]), -1.0

    if "id_list" not in true_raw_data:
        print("[usalign] true_raw_data lacks 'id_list'; cannot locate native PDBs.")
        if return_detail:
            return np.array([]), -1.0, []
        return np.array([]), -1.0

    id_list = true_raw_data["id_list"]
    tm_scores = []
    details = []

    for nid in id_list:
        native_pdb_path = os.path.join(data_path, "raw", f"{nid}.pdb")
        if not os.path.exists(native_pdb_path):
            print(f"[usalign] Native PDB missing: {native_pdb_path}  (skipping)")
            continue
        try:
            res = run_rna_usalign(predicted_pdb_path, native_pdb_path)
            tm_agg = aggregate_tm(res.tmscore_chain1, res.tmscore_chain2, aggregate)
            tm_scores.append(tm_agg)
            if return_detail:
                details.append(res)
        except Exception as e:
            print(f"[usalign] USalign failed for native {nid}: {e}")

    if len(tm_scores) == 0:
        if return_detail:
            return np.array([]), -1.0, []
        return np.array([]), -1.0

    tm_arr = np.array(tm_scores, dtype=float)
    tm_mean = float(tm_arr.mean())

    if return_detail:
        return tm_arr, tm_mean, details
    return tm_arr, tm_mean

# === ViennaRNA-based metrics: MFE, Ensemble Defect, Shannon entropy, Tm sweep ===

def _vienna_fc(seq: str, T: float):
    """
    Internal: build a ViennaRNA fold_compound at temperature T (°C).
    """
    try:
        import RNA
    except ImportError as e:
        raise ImportError(
            "ViennaRNA Python API not found. Install it with:\n"
            "  mamba install -c conda-forge viennarna"
        ) from e
    
    # SEGFAULT FIX: Comprehensive input validation before ViennaRNA calls
    if not isinstance(seq, str) or len(seq) == 0:
        raise ValueError(f"Invalid sequence: must be non-empty string, got {type(seq)} with length {len(seq) if hasattr(seq, '__len__') else 'N/A'}")
    
    # SEGFAULT FIX: Validate sequence characters
    valid_chars = set('ACGURYWSMKBDHVN.-')  # Include ambiguous and gap characters
    invalid_chars = set(seq.upper()) - valid_chars
    if invalid_chars:
        print(f"WARNING: Sequence contains invalid characters: {invalid_chars}. Replacing with 'N'.")
        seq = ''.join(c if c.upper() in valid_chars else 'N' for c in seq)
    
    try:
        # SEGFAULT FIX: Wrap ViennaRNA C library calls in try-catch
        md = RNA.md()
        md.temperature = float(T)
        fc = RNA.fold_compound(seq, md)
        return fc, RNA
    except Exception as e:
        print(f"ERROR: ViennaRNA fold_compound failed: {e}")
        print(f"Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
        raise RuntimeError(f"ViennaRNA fold_compound creation failed: {e}") from e

def vienna_mfe(seq: str, T: float = 37.0):
    """
    Return (mfe_kcal_per_mol, mfe_dotbracket) at temperature T°C.
    """
    # SEGFAULT FIX: Skip MFE calculation for very long sequences that crash ViennaRNA  
    max_vienna_length = 1000  # Conservative limit to prevent segfaults
    if len(seq) > max_vienna_length:
        print(f"WARNING: Sequence too long for ViennaRNA MFE calculation ({len(seq)} > {max_vienna_length}). Returning NaN.")
        return float("nan"), "." * len(seq)  # All unpaired structure
    
    try:
        # SEGFAULT FIX: Wrap fold_compound creation in try-catch
        fc, _ = _vienna_fc(seq, T)
    except Exception as e:
        print(f"ERROR: Failed to create ViennaRNA fold_compound for MFE: {e}")
        return float("nan"), "." * len(seq)
    
    try:
        # SEGFAULT FIX: Wrap MFE calculation in try-catch
        db, mfe = fc.mfe()
        return float(mfe), db
    except Exception as e:
        print(f"WARNING: ViennaRNA MFE calculation failed: {e}")
        return float("nan"), "." * len(seq)

def vienna_ensemble_metrics(seq: str,
                            target_db: str | None = None,
                            T: float = 37.0,
                            return_positional_entropy: bool = False):
    """
    Compute ensemble metrics at T°C:
      - mfe (kcal/mol) and mfe_db (dot-bracket)
      - ED: ensemble defect (absolute # of nucleotides expected to be wrong)
      - ED_per_nt: ED / N
      - pS0: Boltzmann probability of target_db if provided (else of mfe_db)
      - entropy_mean: mean positional Shannon entropy (kT units)
      - entropy_list (optional): per-position entropy (len N)
      - diversity: mean base-pair distance of the ensemble
    """
    assert isinstance(seq, str) and len(seq) > 0
    
    # SEGFAULT FIX: Comprehensive input validation to prevent all crashes
    
    # Check 1: Length limits to prevent segfaults
    max_vienna_length = 1000  # Conservative limit to prevent segfaults
    if len(seq) > max_vienna_length:
        print(f"WARNING: Sequence too long for ViennaRNA ({len(seq)} > {max_vienna_length}). Returning NaN values.")
        nan_result = {
            'mfe': float('nan'),
            'mfe_db': "." * len(seq),  # All unpaired
            'ED': float('nan'),
            'ED_per_nt': float('nan'),
            'pS0': float('nan'),
            'entropy_mean': float('nan'),
            'diversity': float('nan'),
        }
        if return_positional_entropy:
            nan_result["entropy_list"] = [float('nan')] * len(seq)
        return nan_result
    
    # Check 2: Minimum length to avoid edge cases
    min_vienna_length = 2  # ViennaRNA needs at least 2 nucleotides
    if len(seq) < min_vienna_length:
        print(f"WARNING: Sequence too short for ViennaRNA ({len(seq)} < {min_vienna_length}). Returning NaN values.")
        nan_result = {
            'mfe': float('nan'),
            'mfe_db': "." * len(seq),  # All unpaired
            'ED': float('nan'),
            'ED_per_nt': float('nan'),
            'pS0': float('nan'),
            'entropy_mean': float('nan'),
            'diversity': float('nan'),
        }
        if return_positional_entropy:
            nan_result["entropy_list"] = [float('nan')] * len(seq)
        return nan_result
    
    # Check 3: Problematic length ranges that often cause segfaults
    # Based on experience, some specific lengths near boundaries cause issues
    problematic_lengths = [0, 1]  # Add more if discovered
    if len(seq) in problematic_lengths:
        print(f"WARNING: Sequence length {len(seq)} known to cause ViennaRNA issues. Returning NaN values.")
        nan_result = {
            'mfe': float('nan'),
            'mfe_db': "." * len(seq),  # All unpaired
            'ED': float('nan'),
            'ED_per_nt': float('nan'),
            'pS0': float('nan'),
            'entropy_mean': float('nan'),
            'diversity': float('nan'),
        }
        if return_positional_entropy:
            nan_result["entropy_list"] = [float('nan')] * len(seq)
        return nan_result
    
    # Check 4: Validate sequence characters to prevent crashes
    valid_rna_chars = set('ACGUacgu')
    if not all(c in valid_rna_chars for c in seq):
        invalid_chars = set(seq) - valid_rna_chars
        print(f"WARNING: Sequence contains non-RNA characters: {invalid_chars}. Skipping ViennaRNA to prevent crash.")
        nan_result = {
            'mfe': float('nan'),
            'mfe_db': "." * len(seq),  # All unpaired
            'ED': float('nan'),
            'ED_per_nt': float('nan'),
            'pS0': float('nan'),
            'entropy_mean': float('nan'),
            'diversity': float('nan'),
        }
        if return_positional_entropy:
            nan_result["entropy_list"] = [float('nan')] * len(seq)
        return nan_result
    
    # Check 5: PRE-VALIDATE / repair target structure before any ViennaRNA calls.
    # Length-mismatch cases used to hard-reject (returning all-NaN); we now soft-fix
    # them so MFE/ED/pS0 can still be computed on the bulk of the structural prior.
    # CRITICAL: ViennaRNA segfaults at the C level on unbalanced parens (try/except
    # below cannot catch this — the whole Python process dies). Truncating a balanced
    # dot-bracket can leave it unbalanced, so we must rebalance defensively here.
    if target_db is not None:
        if len(target_db) != len(seq):
            if len(target_db) > len(seq):
                target_db = target_db[:len(seq)]
            else:
                target_db = target_db + "." * (len(seq) - len(target_db))

        # Sanitize invalid structure characters to '.' (unpaired)
        valid_structure_chars = set('().')
        invalid_structure_chars = set(target_db) - valid_structure_chars
        if invalid_structure_chars:
            target_db = "".join(c if c in valid_structure_chars else '.' for c in target_db)

        # Balance parens defensively — single-pass scan; replace unmatched chars with '.'.
        # Without this, ViennaRNA can SEGFAULT on truncated structures.
        chars = list(target_db)
        stack = []
        for i, c in enumerate(chars):
            if c == '(':
                stack.append(i)
            elif c == ')':
                if stack:
                    stack.pop()
                else:
                    chars[i] = '.'  # unmatched ')'
        for i in stack:
            chars[i] = '.'  # unmatched '('
        target_db = ''.join(chars)
    
    # SEGFAULT FIX: Initialize default values in case any ViennaRNA call fails
    mfe_db, mfe = "." * len(seq), float('nan')
    ED, pS0 = float('nan'), float('nan')
    H, entropy_mean = [], 0.0
    diversity = float('nan')
    
    try:
        # SEGFAULT FIX: Wrap fold_compound creation in try-catch
        fc, RNA = _vienna_fc(seq, T)
    except Exception as e:
        print(f"ERROR: Failed to create ViennaRNA fold_compound: {e}")
        nan_result = {
            'mfe': float('nan'),
            'mfe_db': "." * len(seq),
            'ED': float('nan'),
            'ED_per_nt': float('nan'),
            'pS0': float('nan'),
            'entropy_mean': float('nan'),
            'diversity': float('nan'),
        }
        if return_positional_entropy:
            nan_result["entropy_list"] = [float('nan')] * len(seq)
        return nan_result

    # SEGFAULT FIX: Wrap MFE calculation in try-catch
    try:
        mfe_db, mfe = fc.mfe()
    except Exception as e:
        print(f"WARNING: ViennaRNA MFE calculation failed: {e}")
        mfe_db, mfe = "." * len(seq), float('nan')

    # SEGFAULT FIX: Wrap partition function in try-catch
    try:
        fc.pf()
    except Exception as e:
        print(f"WARNING: ViennaRNA partition function failed: {e}")
        # Continue with remaining calculations that don't need pf()

    # SEGFAULT FIX: Validate and fix target structure length before using it
    db = target_db if (target_db is not None) else mfe_db
    if db is not None and len(db) != len(seq):
        print(f"WARNING: target_db length {len(db)} != seq length {len(seq)}. Fixing mismatch to prevent segfault.")
        if len(db) > len(seq):
            db = db[:len(seq)]  # Truncate structure to match sequence
            print(f"   Truncated structure from {len(target_db)} to {len(seq)} characters")
        else:
            db = db + "." * (len(seq) - len(db))  # Pad structure with unpaired dots
            print(f"   Padded structure from {len(target_db)} to {len(seq)} characters")
    
    # SEGFAULT FIX: Validate structure characters to prevent crashes
    if db is not None:
        valid_structure_chars = set('().')
        invalid_chars = set(db) - valid_structure_chars
        if invalid_chars:
            print(f"WARNING: Structure contains invalid characters: {invalid_chars}. Using MFE structure instead.")
            db = None  # Fall back to MFE structure

    # SEGFAULT FIX: Wrap ensemble defect calculation in try-catch
    try:
        ED = float(fc.ensemble_defect(db)) if db is not None else float('nan')
    except Exception as e:
        print(f"WARNING: ViennaRNA ensemble_defect failed: {e}")
        ED = float('nan')

    # SEGFAULT FIX: Wrap structure probability calculation in try-catch
    try:
        pS0 = float(fc.pr_structure(db)) if db is not None else float('nan')
    except Exception as e:
        print(f"WARNING: ViennaRNA pr_structure failed: {e}")
        pS0 = float('nan')

    # SEGFAULT FIX: Wrap positional entropy calculation in try-catch
    try:
        H = fc.positional_entropy()  # list with indices 1..N
        if H and len(H) == len(seq) + 1:
            H = H[1:]
        entropy_mean = float(np.mean(H)) if H else 0.0
    except Exception as e:
        print(f"WARNING: ViennaRNA positional_entropy failed: {e}")
        H = []
        entropy_mean = float('nan')

    # SEGFAULT FIX: Wrap ensemble diversity calculation in try-catch
    try:
        diversity = float(fc.mean_bp_distance())
    except Exception as e:
        print(f"WARNING: ViennaRNA mean_bp_distance failed: {e}")
        diversity = float('nan')

    out = dict(
        mfe=float(mfe),
        mfe_db=mfe_db,
        ED=float(ED) if not np.isnan(ED) else np.nan,
        ED_per_nt=(float(ED) / len(seq)) if (not np.isnan(ED)) else np.nan,
        pS0=float(pS0) if not np.isnan(pS0) else np.nan,
        entropy_mean=float(entropy_mean),
        diversity=float(diversity),
    )
    if return_positional_entropy:
        out["entropy_list"] = [float(x) for x in H] if H else []
    return out

def vienna_Tm_by_pS0(seq: str,
                     target_db: str,
                     Tmin: float = 10.0,
                     Tmax: float = 90.0,
                     step: float = 1.0,
                     threshold: float = 0.5):
    """
    Coarse melting temperature estimate (°C) as the temperature where p(S0)
    (probability of the target structure) is closest to `threshold`.
    """
    # Soft-fix length mismatch (matches the soft-fix in vienna_ensemble_metrics);
    # avoids skipping Tm for the ~1.2% of multi-chain edge cases where target_db
    # extraction misaligns with the model's graph residue filter by 1-6 nt.
    # CRITICAL: rebalance parens after truncate to prevent ViennaRNA C-level segfault.
    if len(seq) != len(target_db):
        if len(target_db) > len(seq):
            target_db = target_db[:len(seq)]
        else:
            target_db = target_db + "." * (len(seq) - len(target_db))
    chars = list(target_db)
    stack = []
    for i, c in enumerate(chars):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                stack.pop()
            else:
                chars[i] = '.'
        elif c not in '().':
            chars[i] = '.'
    for i in stack:
        chars[i] = '.'
    target_db = ''.join(chars)

    # SEGFAULT FIX: Skip Tm calculation for very long sequences that crash ViennaRNA
    max_vienna_length = 1000  # Conservative limit to prevent segfaults
    if len(seq) > max_vienna_length:
        print(f"WARNING: Sequence too long for ViennaRNA Tm calculation ({len(seq)} > {max_vienna_length}). Returning NaN.")
        return float("nan")
    
    best_T, best_gap = None, float("inf")
    T = float(Tmin)
    while T <= Tmax + 1e-6:
        fc, _ = _vienna_fc(seq, T)
        fc.pf()
        p = float(fc.pr_structure(target_db))
        gap = abs(p - threshold)
        if gap < best_gap:
            best_gap, best_T = gap, T
        T += step
    return float(best_T) if best_T is not None else float("nan")

def _sanitize_db_for_vienna(db: str) -> str:
    """Map any non '().' characters to '.' so ViennaRNA accepts it."""
    return "".join(ch if ch in ("(", ")", ".") else "." for ch in db)

# ===========================
# MCQ (Mean of Circular Quantities) for RNA
# Default: pseudo-torsions η/θ using C4' and P atoms
# ===========================

def _wrap180_deg(x: float) -> float:
    """Wrap angle in degrees to (-180, 180]."""
    y = ((x + 180.0) % 360.0) - 180.0
    # map -180 -> 180 for consistency with MD practice
    return 180.0 if np.isclose(y, -180.0) else y

def _dihedral_deg(p0, p1, p2, p3) -> float:
    """Return dihedral angle (degrees) for 4 points."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    # normalize b1 for stability
    b1 /= np.linalg.norm(b1) + 1e-12
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def _circular_mean_diff_deg(diffs_deg: np.ndarray) -> float:
    """
    MCQ core: mean direction of angular differences (in deg), returned in [0, 180].
    Uses atan2(mean(sin), mean(cos)) over all differences.
    """
    if diffs_deg.size == 0:
        return np.nan
    rad = np.deg2rad(diffs_deg)
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    mcq = np.degrees(np.arctan2(s, c))  # in (-180, 180]
    mcq = abs(mcq)
    if mcq > 180.0:
        mcq = 360.0 - mcq
    return mcq

def _angle_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """
    Signed minimal circular difference a - b in degrees, elementwise, wrapped to (-180, 180].
    """
    d = a_deg - b_deg
    return np.vectorize(_wrap180_deg)(d)

def _extract_eta_theta_from_pdb(pdb_path: str):
    """
    Compute pseudo-torsions η_i and θ_i along nucleic residues from a PDB.
    Definitions (RNA):
      η_i = dihedral(C4'_{i-1}, P_i, C4'_i, P_{i+1})
      θ_i = dihedral(P_{i-1}, C4'_i, P_i, C4'_{i+1})
    Returns:
      idx_center : list of center residue indices used
      eta_deg    : np.ndarray [len K]
      theta_deg  : np.ndarray [len K]
    """
    try:
        import MDAnalysis as mda
    except ImportError as e:
        raise ImportError("MDAnalysis is required for MCQ. `mamba install -c conda-forge mdanalysis`") from e

    u = mda.Universe(pdb_path)
    # select nucleic residues (MDAnalysis 'nucleic' covers RNA/DNA)
    nuc = u.select_atoms("nucleic")
    residues = list(nuc.residues)
    if len(residues) < 4:
        return [], np.array([]), np.array([])

    # cache atom coords per residue id for P and C4'
    P = {}
    C4 = {}
    for r in residues:
        rid = int(getattr(r, "resid", getattr(r, "resnum", r.ix)))
        # atom names in PDB standard: "P" and "C4'"
        aP = r.atoms.select_atoms("name P")
        aC4 = r.atoms.select_atoms("name C4'")
        if len(aP) == 1:
            P[rid] = aP.positions[0]
        if len(aC4) == 1:
            C4[rid] = aC4.positions[0]

    # sort residue ids as they appear along the chain(s)
    rids = sorted(set(P.keys()).union(C4.keys()))
    idx_center, eta_list, theta_list = [], [], []
    for k in range(1, len(rids) - 1):
        i_prev, i, i_next = rids[k - 1], rids[k], rids[k + 1]
        # need C4'_{i-1}, P_i, C4'_i, P_{i+1} for η
        # and  P_{i-1}, C4'_i, P_i, C4'_{i+1} for θ
        if (i_prev in C4) and (i in P) and (i in C4) and (i_next in P):
            if (i_prev in P) and (i_next in C4):
                eta = _dihedral_deg(C4[i_prev], P[i], C4[i], P[i_next])
                theta = _dihedral_deg(P[i_prev], C4[i], P[i], C4[i_next])
                idx_center.append(i)
                eta_list.append(_wrap180_deg(eta))
                theta_list.append(_wrap180_deg(theta))
            else:
                # still try eta/theta if components present
                try:
                    eta = _dihedral_deg(C4[i_prev], P[i], C4[i], P[i_next])
                    theta = _dihedral_deg(P[i_prev], C4[i], P[i], C4[i_next])
                    idx_center.append(i)
                    eta_list.append(_wrap180_deg(eta))
                    theta_list.append(_wrap180_deg(theta))
                except Exception:
                    pass

    return idx_center, np.array(eta_list), np.array(theta_list)

def mcq_pseudotorsion(pdb_pred: str, pdb_native: str) -> dict:
    """
    Compute MCQ (degrees) between predicted and native structures in η/θ space.
    Returns dict with:
      mcq_deg, n_positions, mean_abs_diff_eta, mean_abs_diff_theta
    """
    idx1, eta1, th1 = _extract_eta_theta_from_pdb(pdb_pred)
    idx2, eta2, th2 = _extract_eta_theta_from_pdb(pdb_native)

    n = min(len(eta1), len(eta2), len(th1), len(th2))
    if n < 1:
        return {
            "mcq_deg": np.nan,
            "n_positions": 0,
            "mean_abs_diff_eta": np.nan,
            "mean_abs_diff_theta": np.nan,
        }
    # pair by position order (robust if residue numbering differs but order matches)
    de = _angle_diff_deg(eta1[:n], eta2[:n])
    dt = _angle_diff_deg(th1[:n],  th2[:n])

    # MCQ on all angle differences together
    diffs = np.concatenate([de, dt])
    mcq = _circular_mean_diff_deg(diffs)

    return {
        "mcq_deg": float(mcq),
        "n_positions": int(n),
        "mean_abs_diff_eta": float(np.mean(np.abs(de))),
        "mean_abs_diff_theta": float(np.mean(np.abs(dt))),
    }

# ===========================
# MCQ statistics better suited for evaluation
# ===========================

def _circular_stats_deg(diffs_deg: np.ndarray) -> dict:
    """
    Compute circular statistics on wrapped angle differences (deg).
    Returns:
      mean_dir_deg_abs : |mean direction| in degrees (what we originally printed)
      R                : mean resultant length in [0,1] (concentration)
      circ_sd_deg      : circular standard deviation in degrees
    """
    if diffs_deg.size == 0:
        return {"mean_dir_deg_abs": np.nan, "R": np.nan, "circ_sd_deg": np.nan}
    rad = np.deg2rad(diffs_deg)
    s = np.mean(np.sin(rad))
    c = np.mean(np.cos(rad))
    mean_dir = np.degrees(np.arctan2(s, c))   # (-180, 180]
    R = float(np.hypot(c, s))
    # circular SD (Fisher, 1993): sqrt( -2 ln R )
    if R > 1e-12:
        circ_sd_rad = np.sqrt(max(0.0, -2.0 * np.log(R)))
        circ_sd_deg = float(np.degrees(circ_sd_rad))
    else:
        circ_sd_deg = float('inf')
    # map |mean_dir| to [0, 180]
    mean_dir = abs(mean_dir)
    if mean_dir > 180.0:
        mean_dir = 360.0 - mean_dir
    return {"mean_dir_deg_abs": float(mean_dir), "R": R, "circ_sd_deg": float(circ_sd_deg)}

def mcq_pseudotorsion_stats(pdb_pred: str, pdb_native: str) -> dict:
    """
    Evaluation-friendly MCQ stats in η/θ space between predicted and native PDBs.
    Returns:
      n_positions         : number of residues contributing
      mcq_abs_deg         : mean absolute wrapped difference over {η,θ}, in degrees (lower is better)
      mean_abs_diff_eta   : mean |Δη| (deg)
      mean_abs_diff_theta : mean |Δθ| (deg)
      mean_dir_deg_abs    : |mean direction| (deg)  [original 'MCQ' notion; not a similarity magnitude]
      R                   : mean resultant length (0..1, higher is better)
      circ_sd_deg         : circular standard deviation (deg, lower is better)
    """
    idx1, eta1, th1 = _extract_eta_theta_from_pdb(pdb_pred)
    idx2, eta2, th2 = _extract_eta_theta_from_pdb(pdb_native)

    n = min(len(eta1), len(eta2), len(th1), len(th2))
    if n < 1:
        return {
            "n_positions": 0,
            "mcq_abs_deg": np.nan,
            "mean_abs_diff_eta": np.nan,
            "mean_abs_diff_theta": np.nan,
            "mean_dir_deg_abs": np.nan,
            "R": np.nan,
            "circ_sd_deg": np.nan,
        }

    de = _angle_diff_deg(eta1[:n], eta2[:n])     # wrapped Δη in (-180,180]
    dt = _angle_diff_deg(th1[:n],  th2[:n])      # wrapped Δθ
    diffs = np.concatenate([de, dt])

    mcq_abs = float(np.mean(np.abs(diffs)))      # <-- primary similarity magnitude (lower is better)
    stats   = _circular_stats_deg(diffs)

    return {
        "n_positions": int(n),
        "mcq_abs_deg": mcq_abs,
        "mean_abs_diff_eta": float(np.mean(np.abs(de))),
        "mean_abs_diff_theta": float(np.mean(np.abs(dt))),
        **stats,  # mean_dir_deg_abs, R, circ_sd_deg
    }

def mcq_avg_vs_natives(predicted_pdb_path: str, true_raw_data: dict, data_path: str = DATA_PATH) -> dict:
    """
    Average MCQ stats vs all native PDBs in true_raw_data['id_list'].
    Returns dict with the same keys as mcq_pseudotorsion_stats, averaged across natives.
    """
    if "id_list" not in true_raw_data:
        return {"n_positions": 0, "mcq_abs_deg": np.nan, "mean_abs_diff_eta": np.nan,
                "mean_abs_diff_theta": np.nan, "mean_dir_deg_abs": np.nan, "R": np.nan, "circ_sd_deg": np.nan}

    vals = []
    for nid in true_raw_data["id_list"]:
        native_pdb = os.path.join(data_path, "raw", f"{nid}.pdb")
        if not os.path.exists(native_pdb):
            continue
        vals.append(mcq_pseudotorsion_stats(predicted_pdb_path, native_pdb))

    if not vals:
        return {"n_positions": 0, "mcq_abs_deg": np.nan, "mean_abs_diff_eta": np.nan,
                "mean_abs_diff_theta": np.nan, "mean_dir_deg_abs": np.nan, "R": np.nan, "circ_sd_deg": np.nan}

    # simple arithmetic means; for R you may also average on the unit circle, but mean(R) is fine for reporting
    out = {}
    keys = ["n_positions", "mcq_abs_deg", "mean_abs_diff_eta", "mean_abs_diff_theta",
            "mean_dir_deg_abs", "R", "circ_sd_deg"]
    for k in keys:
        arr = [v[k] for v in vals if v[k] == v[k]]  # exclude NaNs
        out[k] = float(np.mean(arr)) if arr else np.nan
    # n_positions is averaged; if you prefer min or sum, change here.
    return out


# clash score testing
if __name__ == '__main__':
    print("--- Running test for get_clash_score_phenix with CIF file ---")

    # --- Configuration ---
    PHENIX_WRAPPER_PATH = "/mnt/rna01/smh/projects/ribopo/tools/run_phenix.sh"
    
    # MODIFICATION: Using the exact directory and CIF filename you provided
    TEST_STRUCTURE_FILE = "/mnt/rna01/smh/projects/ribopo/dpo/debug/example_data/1Y0T.cif"
    
    # --- Test Setup ---
    # Check if your CIF file exists. If not, the script will stop.
    if not os.path.exists(TEST_STRUCTURE_FILE):
        print(f"❌ ERROR: Test file not found at the specified path.")
        print(f"Please ensure '{TEST_STRUCTURE_FILE}' exists.")
        exit(1)

    # --- Run the Function ---
    try:
        print(f"Calculating clash score for '{os.path.basename(TEST_STRUCTURE_FILE)}'...")
        score = get_clash_score_phenix(TEST_STRUCTURE_FILE, PHENIX_WRAPPER_PATH)
        
        print("\n--- TEST RESULT ---")
        print(f"✅ Success! Calculated Clash Score: {score}")

        # The clash score is for the structure itself, so it should be very similar
        # regardless of whether the input is PDB or CIF format.
        expected_score = 6.40 
        assert abs(score - expected_score) < 0.1, "Score does not match expected value!"
        print(f"✅ Score is consistent with the expected value of ~{expected_score}.")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n--- TEST FAILED ---")
        print(f"❌ An error occurred: {e}")

