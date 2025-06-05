import faiss
import os
from pathlib import Path

faiss_dir = Path(faiss.__file__).parent
lib_dir = faiss_dir.parent / "lib"
include_dir = faiss_dir / "include"

if not lib_dir.exists():
    candidates = list(faiss_dir.glob("*.so"))
    if candidates:
        lib_path = candidates[0]
    else:
        raise RuntimeError("Could not find faiss library file.")
else:
    lib_candidates = list(lib_dir.glob("libfaiss*.so"))
    if not lib_candidates:
        raise RuntimeError("Could not find libfaiss*.so in faiss lib dir.")
    lib_path = lib_candidates[0]

if not include_dir.exists():
    raise RuntimeError("Could not find include directory in faiss.")

print(f"Exporting FAISS env vars:")
print(f"  PYTHON_FAISS_INCLUDE_DIR = {include_dir}")
print(f"  PYTHON_FAISS_LIBRARY_PATH = {lib_path}")

os.environ["PYTHON_FAISS_INCLUDE_DIR"] = str(include_dir)
os.environ["PYTHON_FAISS_LIBRARY_PATH"] = str(lib_path)
os.environ["PYTHON_FAISS_IS_GPU_ENABLED"] = "ON"
