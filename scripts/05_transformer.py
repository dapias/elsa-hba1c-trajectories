"""Time-aware Transformer (Table 3 row T, Tables S5-S6). Requires PyTorch; see elsa_hba1c/transformer.py.

    python scripts/05_transformer.py --data ../tab --work ../work_local --out ../outputs/paper
    python scripts/05_transformer.py --d-model 32 --tag d32       # robustness runs, Table S6
    python scripts/05_transformer.py --d-model 128 --tag d128
    python scripts/05_transformer.py --layers 3 --tag l3

Needs features.pkl and nurse_long.pkl from 01_prepare_data.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere without installing

from elsa_hba1c.transformer import main

if __name__ == "__main__":
    main()
