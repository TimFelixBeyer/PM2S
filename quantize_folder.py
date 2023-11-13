import argparse
import os
from typing import List, Tuple
from tqdm import tqdm

import numpy as np

from quantize_song import quantize_and_compare

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default="/rhome/beyer/reproduction/PM2S/pm2s/data/asap-dataset-master-pm2s-reproduced")
args = parser.parse_args()


paths: List[Tuple[str, str]] = []

for root, dirs, files in os.walk(args.dataset_root):
    if "midi_score.mid" in files:
        gt_file = os.path.join(root, "midi_score.mid")
        for file in files:
            if file.endswith(".mid") and file != "midi_score.mid" and not file.endswith("_quantized.mid"):
                paths.append((os.path.join(root, file), gt_file))
paths = sorted(paths)

results_pred = {
    "Multi-pitch": [],
    "Voice": [],
    "Meter": [],
    "Value": [],
    "Harmony": [],
    "MV2H": [],
}

results_unquant = {
    "Multi-pitch": [],
    "Voice": [],
    "Meter": [],
    "Value": [],
    "Harmony": [],
    "MV2H": [],
}

for midi, gt in tqdm(paths):
    print(midi)
    try:
        result_pred, result_unquant = quantize_and_compare(midi, gt) or ({}, {})
    except ValueError as e:
        print(e, 'skipping')
        continue
    for k in result_pred:
        results_pred[k].append(result_pred[k])
    for k in result_unquant:
        results_unquant[k].append(result_unquant[k])

print("MV2H ---------------------")
for k in results_pred:
    print(k, np.mean(results_pred[k]))
print("Unquantized --------------")
for k in results_unquant:
    print(k, np.mean(results_unquant[k]))
