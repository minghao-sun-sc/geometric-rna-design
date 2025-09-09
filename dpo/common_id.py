# dpo/common_id.py
import os, re
from collections import Counter
from typing import Iterable, Any, Dict, Optional

def canonicalize_id(s: str) -> str:
    s = os.path.basename(str(s))
    s = re.sub(r"\.(pdb|cif|mmcif)$", "", s, flags=re.IGNORECASE)
    parts = s.split("_")
    if len(parts) >= 3:
        chain = parts[2].split("-")[0]
        s = "_".join([parts[0], parts[1], chain])
    else:
        s = s.split("-")[0]
    s = re.sub(r"_+", "_", s)
    return s.strip()

def canonical_from_id_list(id_list: Iterable[str]) -> str:
    c = Counter()
    for it in (id_list or []):
        cid = canonicalize_id(it)
        if cid:
            c[cid] += 1
    return c.most_common(1)[0][0] if c else ""

CANDIDATE_ID_KEYS = [
    "backbone_id", "backbone", "target", "target_id",
    "structure_id", "graph_id", "pdb_model_chain",
    "pdb_chain", "pdb_id", "id", "pdb_file"
]

CANDIDATE_INDEX_KEYS = [
    "index", "graph_index", "idx", "data_index", "backbone_index"
]

def extract_index_from_pair(p: Dict[str, Any]) -> Optional[int]:
    for k in CANDIDATE_INDEX_KEYS:
        if k in p and isinstance(p[k], int):
            return p[k]
    return None

def extract_backbone_id_from_pair(p: Dict[str, Any]) -> str:
    # 1) flat keys, including pdb_file
    for k in CANDIDATE_ID_KEYS:
        if k in p and p[k]:
            return canonicalize_id(str(p[k]))
    # 2) nested meta dict
    meta = p.get("meta", {})
    if isinstance(meta, dict):
        for k in CANDIDATE_ID_KEYS:
            if k in meta and meta[k]:
                return canonicalize_id(str(meta[k]))
    # 3) list of ids
    for k in ["id_list", "src_ids", "ids", "graph_ids"]:
        if k in p and isinstance(p[k], (list, tuple)) and p[k]:
            return canonical_from_id_list(p[k])
    return ""
