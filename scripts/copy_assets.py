"""
Helper script to copy pre-trained binary assets (hand_landmarker.task and pca.joblib)
from reference path into the active workspace.
"""

import os
import shutil

SOURCE_DIR = r"d:\Cognizant\palm-pay"
TARGET_DIR = r"d:\palmmpayy"

ASSETS = ["hand_landmarker.task", "pca.joblib", "mobilenet_v3_palm.pth"]

def copy_assets():
    print(f"Copying model assets from {SOURCE_DIR} to {TARGET_DIR}...")
    for asset in ASSETS:
        src = os.path.join(SOURCE_DIR, asset)
        dst = os.path.join(TARGET_DIR, asset)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size = os.path.getsize(dst)
            print(f"  [✓] Copied {asset} ({size:,} bytes)")
        else:
            print(f"  [!] Asset not found at {src}")

if __name__ == "__main__":
    copy_assets()
