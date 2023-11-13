"""Simple script to quantize a single MIDI file and show MV2H evaluations."""
import argparse
import os
import pprint
import sys
from subprocess import CalledProcessError
from typing import Tuple

import numpy as np
import pretty_midi

sys.path.append(os.getcwd() + "/dev")

from dev.test import mv2h, predict


def quantize_and_compare(midi_path, gt_path=None) -> Tuple[dict[str, float], dict[str, float]] | None:
    midi_instruments = pretty_midi.PrettyMIDI(midi_path).instruments
    note_seq = np.array(
        [
            (n.pitch, n.start, n.end - n.start, n.velocity)
            for ins in midi_instruments
            for n in ins.notes
        ]
    )
    predict(note_seq, midi_path[:-4] + "_quantized.mid")
    print("MV2H ---------------------")
    if gt_path is None:
        print(mv2h(midi_path, midi_path[:-4] + "_quantized.mid"))
        return
    try:
        result_pred = mv2h(gt_path, midi_path[:-4] + "_quantized.mid")
        pprint.pprint(result_pred, sort_dicts=False)
    except CalledProcessError as e:
        print(e, 'skipping')
        result_pred = {}

    try:
        print("Unquantized --------------")
        result_unquant = mv2h(gt_path, midi_path)
        pprint.pprint(result_unquant, sort_dicts=False)
        print("Delta --------------------")
        result_delta = {k: result_pred[k]-result_unquant[k] for k in result_pred}
        pprint.pprint(result_delta, sort_dicts=False)
    except CalledProcessError as e:
        print(e, 'skipping')
        result_unquant = {}
    # os.remove(midi_path[:-4] + "_quantized.mid")

    if result_pred == {} and result_unquant == {}:
        raise ValueError("Both results are empty")
    return result_pred, result_unquant



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, default="eval/Jewel.mid")
    parser.add_argument("--gt", type=str, default=None)
    args = parser.parse_args()
    quantize_and_compare(args.midi, args.gt)


