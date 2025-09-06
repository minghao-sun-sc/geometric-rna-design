# dpo/env_bootstrap.py
import os
from pathlib import Path
from typing import Optional, Tuple

def _find_env_file(start: Path) -> Optional[Path]:
    """Search for .env from start up to filesystem root."""
    cur = start
    for _ in range(6):  # plenty for our tree depth
        cand = cur / ".env"
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

def _manual_load_env(env_path: Path):
    with env_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = _strip_quotes(v.strip())
            if k and (os.getenv(k) is None):
                os.environ[k] = v

def _try_dotenv_load(env_path: Path) -> bool:
    try:
        import dotenv  # optional dependency
        dotenv.load_dotenv(env_path, override=False)
        return True
    except Exception:
        return False

def _alias(src_key: str, dst_key: str):
    """If dst is missing and src exists, copy src->dst."""
    if os.getenv(dst_key) is None and os.getenv(src_key) is not None:
        os.environ[dst_key] = os.environ[src_key]

def _prepend_path(dirpath: Path):
    p = os.environ.get("PATH", "")
    d = str(dirpath)
    if d not in p.split(":"):
        os.environ["PATH"] = f"{d}:{p}"

def bootstrap_env():
    # Locate project root (…/offline-dpo) from this file
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # offline-dpo/
    env_path = _find_env_file(project_root)

    if env_path is not None:
        loaded = _try_dotenv_load(env_path)
        if not loaded:
            _manual_load_env(env_path)

    # Map common aliases expected by src/*
    _alias("ETERNAFOLD", "ETERNAFOLD_PATH")
    _alias("X3DNA", "X3DNA_PATH")

    # Ensure x3dna bin is on PATH
    x3 = os.getenv("X3DNA_PATH")
    if x3:
        _prepend_path(Path(x3) / "bin")

    if os.getenv("DPO_ENV_DEBUG") == "1":
        print(f"[env] .env path: {env_path}")
        print(f"[env] ETERNAFOLD={os.getenv('ETERNAFOLD')}")
        print(f"[env] ETERNAFOLD_PATH={os.getenv('ETERNAFOLD_PATH')}")
        print(f"[env] X3DNA={os.getenv('X3DNA')}")
        print(f"[env] X3DNA_PATH={os.getenv('X3DNA_PATH')}")
        print(f"[env] PATH has x3dna bin? {str(Path(os.getenv('X3DNA_PATH',''))/ 'bin') in os.environ.get('PATH','')}")
