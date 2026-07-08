"""Train ASPP Flow Matching on CKMImageNet-Beijing.

This entry point lives in ASPP/ to match the existing project structure:
df_aspp.py + main_*.py + interfence_*.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ASPP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ASPP_DIR.parent
CKM_BEIJING_DIR = PROJECT_ROOT / "ckm_beijing"

if str(ASPP_DIR) not in sys.path:
    sys.path.insert(0, str(ASPP_DIR))
if str(CKM_BEIJING_DIR) not in sys.path:
    sys.path.insert(0, str(CKM_BEIJING_DIR))

from fm_aspp import FlowMatchingModel 
from train_aspp_beijing import main as train_main 

if __name__ == "__main__":
    train_main(FlowMatchingModel)
