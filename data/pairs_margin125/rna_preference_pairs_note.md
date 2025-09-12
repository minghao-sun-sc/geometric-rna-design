Here is the preference pairs constructed (`dpo_pairs_margin125.json`). 

(Note: this file is actually modified from previous version, with 0.125*std margin)

The preference pairs will be used for offline reinforcement learning (DPO), for our RL-based RNA inverse folding project. 

Details of construction of the preference pairs (file name: `dpo_pairs_margin125.json`):

Note: currently, this preference pairs are constructed for the whole sequences, not residue-level. Let's try to use this residue-level preference pairs and see if it works.

Step 1: RNA inverse folding models to get candidate sequences

1. Use gRNAde model, to the inverse folding for RNA single chain structures
2. Generate 10 candidate sequences for each RNA single chain
3. For each RNA candidate, calculate:
   1. RMSD, with the ground truth, measured by RhoFold+ (A RNA structure prediction tool)
   2. pLDDT prediction quality score. Measured by RhoFold+ 
   3. MFE (Minimum Free Energy), calculated by ViennaRNA

The examples of the saved preference pairs data (dpo_pairs_margin125.json):
```
[
  {
    "pdb_file": "./data/raw/6ZU1_1_AW.pdb",
    "winner_seq": "UGGCGGGGGAGCAGCCUGGAGCUCGUCGGGUCAUAACCCGAAGAUCGUCGGCAAAUCCGGCCCGGCCAGCCA",
    "loser_seq": "GCCGGGGGGGGCAACCUGGAGCCCGUCGGGUCAUAAACCGAAGAUUGUCGGCAAAUCCGGCCCCCGGAACCA",
    "winner_metrics": {
      "mfe": -30.700000762939453,
      "rmsd": 3.8264896900638927,
      "plddt": 0.708327054977417
    },
    "loser_metrics": {
      "mfe": -31.200000762939453,
      "rmsd": 11.82980343302432,
      "plddt": 0.47949546575546265
    }
  },
  {
    "pdb_file": "./data/raw/6ZU1_1_AW.pdb",
    "winner_seq": "UGGCGGGGGAGCAGCCUGGAGCUCGUCGGGUCAUAACCCGAAGAUCGUCGGCAAAUCCGGCCCGGCCAGCCA",
    "loser_seq": "GCCGGGGGGAGCAGCCCGGAGCUCGUCGGGUGAUAACCCCAAGAUCGCGGGCAAAUCCCUCCCCCGGAACCA",
    "winner_metrics": {
      "mfe": -30.700000762939453,
      "rmsd": 3.8264896900638927,
      "plddt": 0.708327054977417
    },
    "loser_metrics": {
      "mfe": -34.29999923706055,
      "rmsd": 19.778674947360773,
      "plddt": 0.6369426250457764
    }
  },
  {
    "pdb_file": "./data/raw/6ZU1_1_AW.pdb",
    "winner_seq": "CACGGGGGGUGCAGCCUGGAGCACGUCGGGUCAUAACCCGAAGAUCGUCGGCAAAUCCGGCCCCCGUAACCA",
    "loser_seq": "GCCGGGGGGGGCAACCUGGAGCCCGUCGGGUCAUAAACCGAAGAUUGUCGGCAAAUCCGGCCCCCGGAACCA",
    "winner_metrics": {
      "mfe": -30.0,
      "rmsd": 3.3904741119173276,
      "plddt": 0.740713357925415
    },
    "loser_metrics": {
      "mfe": -31.200000762939453,
      "rmsd": 11.82980343302432,
      "plddt": 0.47949546575546265
    }
  },
  ......
    {
    "pdb_file": "./data/raw/8AGW_1_x.pdb",
    "winner_seq": "ACCUAUUUAGCACAGCUGAGUGCACCGAGCUUCCGUCUCGGGGGUCGCUGGUUCGAUUCCGGCACUAGGUACCU",
    "loser_seq": "UGGCCCGUGGCGUAGUGGAGCACGCUGUCCGCGUGUGACAGUCGUCGUCGGUUCGAAUCCGGCGAGGGGAGCCA",
    "winner_metrics": {
      "mfe": -24.799999237060547,
      "rmsd": 3.185251546854911,
      "plddt": 0.7820842862129211
    },
    "loser_metrics": {
      "mfe": -31.700000762939453,
      "rmsd": 13.168445019930132,
      "plddt": 0.4653688967227936
    }
  },
  ......
```

Step 2: Clean the data

- Delete the sequences with RMSD >= 128 (high structure bias)
- Delete the sequences with pLDDT < 0.30 (low prediction quality)

The statistics after data cleaning:

```
Original PDB: 7763
The PDB data after cleaning: 7072
Number of Original Seqs: 77090
Sequence deleted: 8374
Sequences after cleaning: 68716
```

Step 3: The correlation of the metrics

Calculate the pair-wise Spearman correlation of MFE, RMSD, pLDDT

```
Spearman (N=68716)

mfe_rmsd → corr = 0.0414, p = 1.6381e-27
mfe_plddt → corr = -0.0484, p = 6.7336e-37
rmsd_plddt → corr = -0.7019, p = 0.0000e+00
```

Note: 

- Strong negative correlation between pLDDT and RMSD
- MFE almost have no correlation with RMSD and pLDDT. MFE can be a subsidiary ranking metric.

Step 4: Statistics and Visualization

```
mfe rmsd plddt
count 68716.000000 68716.000000 68716.000000
mean -24.416626 9.025253 0.692498
std 15.307580 7.890024 0.147140
min -141.300003 1.125736 0.300006
25% -35.000000 2.991599 0.614751
50% -25.299999 5.593675 0.735734
75% -11.400000 13.839653 0.802008
max 0.000000 89.162239 0.926649
```

Step 5: Construct the preference pairs

1. Defining the Preference (as winner/preference seqs):
   1. RMSD < 8
   2. pLDDT > 0.7

2. Construct the pairs randomly:
   1. Use RMSD / pLDDT 0.125 * std as the gap threshold, to decide winner or loser
   2. For MFE, we only requires that MFE_winner < MFE_loser (winner with better MFE metric)
   3. Combine the pairs randomly

3. Output the JSON format results (ribopo_pairs.json)
   1. Totally, 27133 preference pairs are constructed





