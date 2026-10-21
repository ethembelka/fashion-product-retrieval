"""fashret: consumer-to-shop fashion retrieval on DeepFashion2 with open-set rejection."""

# Windows DLL conflict: loading pyarrow.dataset AFTER torch (CUDA) crashes the process (access violation
# in arrow's DLL init). Importing it here, before any torch import, is safe. Scripts therefore import
# `fashret` before `torch`.
try:
    import pyarrow.dataset  # noqa: F401
except Exception:  # pyarrow optional on CPU-only evaluation boxes
    pass

__version__ = "0.1.0"
