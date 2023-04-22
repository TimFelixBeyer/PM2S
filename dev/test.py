import argparse
import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], ".."))
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
from data.dataset_base import BaseDataset

from pm2s.constants import *
from pm2s.constants import keyName2Number, tolerance
from pm2s.features.beat import RNNJointBeatProcessor
from pm2s.features.hand_part import RNNHandPartProcessor
from pm2s.features.key_signature import RNNKeySignatureProcessor
from pm2s.features.quantisation import RNNJointQuantisationProcessor
from pm2s.features.time_signature import RNNTimeSignatureProcessor

MV2H_BIN = "/rhome/beyer/reproduction/MV2H/bin"


def mv2h(midi_pred, midi_gt):
    """Calls the java implementation of MV2H, which is equivalent to the following commands:
    $ java -cp MV2H_BIN mv2h.tools.Converter -i midi_gt >midi_gt.conv.txt
    $ java -cp MV2H_BIN mv2h.tools.Converter -i midi_pred >midi_pred.conv.txt
    $ java -cp MV2H_BIN mv2h.Main -g midi_gt.conv.txt -t midi_pred.conv.txt
    $ rm midi_gt.conv.txt midi_pred.conv.txt
    """
    try:
        subprocess.run(
            [
                "java",
                "-cp",
                MV2H_BIN,
                "mv2h.tools.Converter",
                "-i",
                midi_gt,
                "-o",
                midi_gt + ".conv.txt",
            ]
        )
        subprocess.run(
            [
                "java",
                "-cp",
                MV2H_BIN,
                "mv2h.tools.Converter",
                "-i",
                midi_pred,
                "-o",
                midi_pred + ".conv.txt",
            ]
        )
        result = subprocess.check_output(
            [
                "java",
                "-cp",
                MV2H_BIN,
                "mv2h.Main",
                "-g",
                midi_gt + ".conv.txt",
                "-t",
                midi_pred + ".conv.txt",
            ],
            universal_newlines=True,
        )
    finally:
        os.remove(midi_gt + ".conv.txt")
        os.remove(midi_pred + ".conv.txt")
    return result


def eval(args):
    processor_beat = RNNJointBeatProcessor(
        "../_model_state_dicts/beat/RNNJointBeatModel.pth"
    )
    processor_beat._model.eval()
    processor_beat._model.to("cuda")
    processor_quantisation = RNNJointQuantisationProcessor(
        "../_model_state_dicts/quantisation/RNNJointQuantisationModel.pth",
        "../_model_state_dicts/beat/RNNJointBeatModel.pth",
    )
    processor_quantisation._model.eval()
    processor_quantisation._model.to("cuda")

    processor_hand = RNNHandPartProcessor(
        "../_model_state_dicts/hand_part/RNNHandPartModel.pth"
    )
    processor_hand._model.eval()
    processor_hand._model.to("cuda")

    processor_time_sig = RNNTimeSignatureProcessor(
        "../_model_state_dicts/time_signature/RNNTimeSignatureModel.pth"
    )
    processor_time_sig._model.eval()
    processor_time_sig._model.to("cuda")
    processor_key_sig = RNNKeySignatureProcessor(
        "../_model_state_dicts/key_signature/RNNKeySignatureModel.pth"
    )
    processor_key_sig._model.eval()
    processor_key_sig._model.to("cuda")


    def predict(note_seq: np.ndarray):
        # beats = processor_beat.process(note_seq)
        beats, onset_positions, note_values = processor_quantisation.process(note_seq)
        hand_parts = processor_hand.process(note_seq)
        time_signatures = processor_time_sig.process(note_seq)
        time_signatures = [(ts[0], *ts[1].split("/")) for ts in time_signatures]
        key_signatures = processor_key_sig.process(note_seq)
        key_signatures = [(ks[0], keyName2Number[ks[1]]) for ks in key_signatures]
        annotations = {
            "onsets_musical": onset_positions,
            "note_value": note_values,
            "hands": hand_parts,
            "time_signatures": time_signatures,
            "key_signatures": key_signatures,
            "beats": [0, *beats], # to align with the gt, we have to add one beat at t=0
        }
        # Create PrettyMIDI object using the above data
        return quantize(note_seq, annotations, "/tmp/test.mid")

    def quantize(note_seq, annotations, save_path, gt_midi=None):
        onset_positions = annotations["onsets_musical"]
        note_values = annotations["note_value"]
        hand_parts = annotations["hands"]
        time_signatures = annotations["time_signatures"]
        key_signatures = annotations["key_signatures"]
        beats = annotations["beats"]
        # Get model output data
        # onset positions
        # note values
        # dict_keys(['beats', 'downbeats', 'time_signatures', 'key_signatures', 'onsets_musical', 'note_value', 'hands'])
        # Create PrettyMIDI object using the above data
        midi = pretty_midi.PrettyMIDI()
        for ts in time_signatures:
            ts, num, denom = ts
            midi.time_signature_changes.append(
                pretty_midi.TimeSignature(int(num), int(denom), ts)
            )
        for ks in key_signatures:
            midi.key_signature_changes.append(
                pretty_midi.KeySignature(int(ks[1]), ks[0])
            )
        instrument_left = pretty_midi.Instrument(program=0, name="Piano left")
        instrument_right = pretty_midi.Instrument(program=0, name="Piano right")
        errors = []
        for i, (pitch, onset, length, velocity) in enumerate(note_seq):
            # The quantized onset is placed at start=(beats[i] + ibi * musical onset)
            # where ibi = beats[i+1] - beats[i]
            # We select i to minimize abs(start-onset) + abs(end-onset-length).
            def quantize_note(beats, onset_position, note_value, idx):
                if gt_midi is None:
                    if idx + 1 < len(beats):
                        ibi = beats[idx + 1] - beats[idx]
                    else:
                        ibi = beats[idx] - beats[idx - 1]
                    start = beats[idx] + ibi * (onset_position % 1)
                else:
                    if idx + 1 < len(beats):
                        ibi = gt_midi.time_to_tick(beats[idx + 1]) - gt_midi.time_to_tick(beats[idx])
                    else:
                        ibi = gt_midi.time_to_tick(beats[idx]) - gt_midi.time_to_tick(beats[idx - 1])
                    start = gt_midi.time_to_tick(beats[idx]) + ibi * (onset_position % 1)

                    start = gt_midi.tick_to_time(round(start))

                # Step forward from beat to beat to find the end of the note
                duration = 0
                remaining = 1 - (onset_position % 1)
                while note_value > 1 and idx + 1 < len(beats):
                    if gt_midi is None:
                        ibi = beats[idx+1] - beats[idx]
                    else:
                        ibi = gt_midi.time_to_tick(beats[idx + 1]) - gt_midi.time_to_tick(beats[idx])
                    duration += ibi * remaining
                    note_value -= remaining
                    remaining = 1
                    idx += 1
                if idx + 1 == len(beats): # at end of song, but not done
                    duration += ibi * note_value
                else:
                    if gt_midi is None:
                        duration += (beats[idx+1] - beats[idx]) * note_value
                    else:
                        duration += (gt_midi.time_to_tick(beats[idx+1]) - gt_midi.time_to_tick(beats[idx])) * note_value
                if gt_midi is None:
                    end = start + duration
                else:
                    end = gt_midi.tick_to_time(round(gt_midi.time_to_tick(start) + duration))

                error = abs(start-onset)# + abs(end-onset-length)
                return start, end, error

            idxs = np.abs(beats - onset).argsort()[:4]
            error = float("inf")
            for idx in idxs:
                new_start, new_end, new_error = quantize_note(beats, onset_positions[i], note_values[i], idx)
                if new_error < error:
                    start, end, error = new_start, new_end, new_error
            # Insert the quantized note into the PrettyMIDI object
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=start,
                end=end
            )
            errors.append([abs(start-onset), abs(end-onset-length)])
            # print("-"*20)
            # print(f"old: {onset:03.4f} {onset+length:03.4f}  |  {onset_positions[i] % 1} + {note_values[i]}")
            # print(f"new: {note.start:03.4f} {note.end:03.4f}  | {abs(start-onset):03.4f} {abs(end-onset-length):03.4f}")

            if hand_parts is None or hand_parts[i] == 0:
                instrument_left.notes.append(note)
            else:
                instrument_right.notes.append(note)
        # Plot the errors
        errors = np.array(errors)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(errors[:, 0])
        plt.title("Onset error")
        plt.subplot(1, 2, 2)
        plt.plot(errors[:, 1])
        plt.title("Offset error")
        plt.savefig(f"errors{time.time()}_{gt_midi is not None}.pdf")
        plt.close()

        midi.instruments.append(instrument_left)
        midi.instruments.append(instrument_right)
        midi.write(save_path)
        return save_path

    metrics = {
        "Multi-pitch": [],
        "Voice": [],
        "Meter": [],
        "Value": [],
        "Harmony": [],
        "MV2H": [],
    }

    dataset = BaseDataset(args.workspace, "test")
    for idx in range(len(dataset)):
        print(f"Evaluating {idx}")
        # Load inputs
        row = dataset._sample_row(idx)
        note_sequence, annotations = dataset._load_data(row)
        midi_pred = predict(note_sequence)

        # Load ground truth
        annotations["onsets_musical"] = np.round(annotations["onsets_musical"] * N_per_beat) / N_per_beat
        annotations["note_value"] = np.round(annotations["note_value"] * N_per_beat) / N_per_beat
        annotations["note_value"][annotations["note_value"] == 0] = 1 / N_per_beat
        midi_gt = quantize(note_sequence, annotations, "/tmp/test_gt.mid", pretty_midi.PrettyMIDI(row["midi_perfm"]))
        # midi_gt = row["midi_perfm"]
        # Evaluate
        result = mv2h(midi_pred, midi_gt)
        for line in result.splitlines():
            key, value = line.split(": ")
            metrics[key].append(float(value))
        print(result)

    for key, value in metrics.items():
        print(key, np.mean(value), "+/-", np.std(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")

    parser.add_argument(
        "--workspace",
        type=str,
        default="/rhome/beyer/reproduction/PM2S",
        help="Workspace directory.",
    )

    args = parser.parse_args()

    eval(args)
