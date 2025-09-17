# Environment Setup Guide

## Mamba Env

```bash
# Mamba Installation
# 1. Create a personal directory in /tmp and move into it
mkdir /tmp/smh
cd /tmp/smh

# 2. Download and run the installer for that location
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /tmp/smh/miniforge3

# 3. Initialize your shell from the new location
/tmp/smh/miniforge3/bin/conda init bash
```

## gRNAde Env

```bash
# Clone gRNAde repository
cd ~  # change this to your prefered download location
git clone https://github.com/chaitjo/geometric-rna-design.git
cd geometric-rna-design

# Install mamba (a faster conda)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc
# You may also use conda or virtualenv to create your environment

# Create new environment and activate it
mamba create -n rna python=3.10
mamba activate rna

# For personal usage, please ignore
mamba install gxx_linux-64

# Install Pytorch on Nvidia GPUs (ensure appropriate CUDA version for your hardware)
# mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c conda-forge -c nvidia
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge

# Install Pytorch Geometric (ensure matching torch + CUDA version to PyTorch)
uv pip install torch_geometric
# Should build by suitable torch version in the last step
# torch_scatter needs PyTorch to be already installed
# For torch-2.0.1; Please use torch series compatiable with the torch version
uv pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.0.1+cu121.html
# uv pip install torch-scatter torch-cluster --no-build-isolation -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
# mamba install jupyterlab matplotlib seaborn pandas biopython biotite -c conda-forge -y
uv pip install jupyterlab matplotlib seaborn pandas biopython biotite
uv pip install wandb gdown pyyaml ipdb python-dotenv tqdm cpdb-protein torchmetrics einops ml_collections mdanalysis MDAnalysisTests draw_rna arnie rna-tools

# For thermostability calculation
mamba install -c bioconda usalign viennarna cd-hit -y
mamba install networkx==2.8

# Create isolated env lddt_env for the lDDT
mamba create -n lddt_env python=3.10 bioconda::openstructure biopython -y

# Check if the gRNAde env was configured successfully
python gRNAde.py --pdb_filepath tutorial/demo_data/4FE5_1_B.pdb --output_filepath tutorial/outputs/demo_output.fasta --split das --max_num_conformers 1 --n_samples 16 --temperature 0.5
```

## RhoFold Protocol Env

With the updated installation with `uv` at `external/rhofold_protocol`:

```bash
cd external/rhofold_protocol/

mamba env create -f environment.yml

mamba activate rhofold_protocol

uv pip install -r requirements.txt
```

Test the RhoFold env configuration:

```bash
python rhofold/inference.py --input_fasta ./data/rhofold/3owz_A/3owz_A.fasta --input_msa ./data/rhofold/3owz_A/3owz_A.afa --output_dir ./results/rhofold/3owz_A --device cuda:0 

python scripts/parse_plddt.py --npz_path ./results/rhofold/3owz_A/results.npz
```

Example Results from the RhoFold:

```bash
(rhofold_protocol) [smh@zgpuA1003 rhofold_protocol]$ python rhofold/inference.py --input_fasta ./data/rhofold/3owz_A/3owz_A.fasta --input_msa ./data/rhofold/3owz_A/3owz_A.afa --output_dir ./results/rhofold/3owz_A --device cuda:0 
2025-08-29 09:31:38,516 - INFO: Constructing RhoFold
2025-08-29 09:31:39,290 - INFO:     loading ./checkpoints/rhofold_pretrained_params.pt
2025-08-29 09:31:39,941 - INFO: Input_fas ./data/rhofold/3owz_A/3owz_A.fasta
2025-08-29 09:31:39,941 - INFO: Input MSA path: ./data/rhofold/3owz_A/3owz_A.afa
2025-08-29 09:31:39,941 - INFO: Started RhoFold Inference
2025-08-29 09:31:39,986 - INFO:     Inference using device cuda:0
2025-08-29 09:31:47,874 - INFO:     Export PDB file to ./results/rhofold/3owz_A/unrelaxed_model.pdb
2025-08-29 09:31:47,875 - INFO: Finished RhoFold Inference in 7.933 seconds
2025-08-29 09:31:47,875 - INFO: Started Amber Relaxation : 1000 iterations
2025-08-29 09:31:47,875 - INFO:     AmberRelaxation: Using GPU
2025-08-29 09:32:03,816 - INFO:     Minimizing ...
2025-08-29 09:34:33,745 - INFO:     Energy at Minima is -492048.232 kcal/mol
2025-08-29 09:34:33,959 - INFO:     Export PDB file to ./results/rhofold/3owz_A/relaxed_1000_model.pdb
2025-08-29 09:34:33,960 - INFO: Finished Amber Relaxation : 1000 iterations in 166.085 seconds

(rhofold_protocol) [smh@zgpuA1003 rhofold_protocol]$ python scripts/parse_plddt.py --npz_path ./results/rhofold/3owz_A/results.npz
mean pLDDT = 0.8479894995689392
```

RhoFold Inference without MSA:

```bash
python rhofold/inference.py --input_fasta ./data/rhofold/3owz_A/3owz_A.fasta --output_dir ./results/rhofold/3owz_A_no_msa --single_seq_pred True --device cuda:0

(rhofold_protocol) [smh@zgpuA1003 rhofold_protocol]$ python rhofold/inference.py --input_fasta ./data/rhofold/3owz_A/3owz_A.fasta --output_dir ./results/rhofold/3owz_A_no_msa --single_seq_pred True --device cuda:0
2025-08-29 23:41:56,502 - INFO: Constructing RhoFold
2025-08-29 23:41:57,279 - INFO:     loading ./checkpoints/rhofold_pretrained_params.pt
2025-08-29 23:41:58,002 - INFO: Input_fas ./data/rhofold/3owz_A/3owz_A.fasta
2025-08-29 23:41:58,003 - INFO: The model will use the single query sequence only. Setting the MSA path to the input fasta file.
2025-08-29 23:41:58,003 - INFO: Started RhoFold Inference
2025-08-29 23:41:58,047 - INFO:     Inference using device cuda:0
2025-08-29 23:42:02,189 - INFO:     Export PDB file to ./results/rhofold/3owz_A_no_msa/unrelaxed_model.pdb
2025-08-29 23:42:02,189 - INFO: Finished RhoFold Inference in 4.186 seconds
2025-08-29 23:42:02,189 - INFO: Started Amber Relaxation : 1000 iterations
2025-08-29 23:42:02,189 - INFO:     AmberRelaxation: Using GPU
2025-08-29 23:42:17,972 - INFO:     Minimizing ...
2025-08-29 23:45:15,452 - INFO:     Energy at Minima is -573412.496 kcal/mol
2025-08-29 23:45:15,622 - INFO:     Export PDB file to ./results/rhofold/3owz_A_no_msa/relaxed_1000_model.pdb
2025-08-29 23:45:15,624 - INFO: Finished Amber Relaxation : 1000 iterations in 193.434 seconds
(rhofold_protocol) [smh@zgpuA1003 rhofold_protocol]$ python scripts/parse_plddt.py --npz_path ./results/rhofold/3owz_A_no_msa/results.npz
mean pLDDT = 0.7739496231079102
```

RhoFold Inference with MSA:

```bash
# Run the MSA Search
# At the DPO-RNA Project
/mnt/dna01/library2/rhofold_protocol/rmsa/rMSA.pl input/test_examples/1EBQ -outdir=input/test_examples/test_outputs/ -cpu=8

mamba activate rhofold_protocol

cd external/rhofold_protocol

python rhofold/inference.py --input_fasta input/test_examples/1EBQ --input_msa input/test_examples/test_outputs/1EBQ.a3m --output_dir input/test_examples/test_outputs/msa_examples/1EBQ --device cuda:0

python scripts/parse_plddt.py --npz_path input/test_examples/test_outputs/results.npz
```

## Protenix-Mini

```bash
uv pip install protenix

uv pip uninstall deepspeed
uv pip install deepspeed==0.12.6
uv pip install gemmi==0.6.7 pdbeccdutils==0.8.6

cd external/Protenix

# ensure `release_data/ccd_cache/components.cif` or run:
python scripts/gen_ccd_cache.py -c release_data/ccd_cache/ -n [num_cpu]

protenix predict --input input/test_examples/protenix_example.json --out_dir input/test_examples/test_outputs --model_name "protenix_mini_esm_v0.5.0" --use_msa false

# protenix predict --input external/Protenix/examples/example.json --out_dir ./tmp/output --model_name "protenix_mini_esm_v0.5.0" --use_msa false

CUDA_VISIBLE_DEVICES=0 \
python scripts/protenix_predict_single.py predict \
  --input ./tmp/testjson/ \
  --out_dir ./tmp/output_minimal \
  --model_name "protenix_mini_esm_v0.5.0" \
  --use_msa false
```

Note, in the input of the Protenix-Mini, entity type must be proteinChain, dnaSequence, rnaSequence, ligand or ion.

Re-configuration of the env

1) Remove the uv/pip versions
uv pip uninstall torch torch-cluster torch-geometric torch-scatter torchaudio torchvision

2) Remove the mamba versions
mamba uninstall pytorch pytorch-cuda torchaudio torchvision


3) Step 2.1: Reinstall the PyTorch stack for CUDA 12.1
uv pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

4) Step 2.2: Reinstall the matching PyG libraries
uv pip install torch_scatter torch_cluster torch_geometric -f https://data.pyg.org/whl/torch-2.3.1+cu121.html



## Eval
The Hamming distance requirement is very strict - it needs to be ≥ 80% of the sequence length. For 20nt, we need ≥16 differences, and for 22nt we need ≥17. 



