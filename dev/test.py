import argparse
import os
import sys

sys.path.insert(0, os.path.join(sys.path[0], ".."))
import subprocess
import numpy as np
import pretty_midi
from data.dataset_base import BaseDataset

from pm2s.constants import keyName2Number, N_per_beat
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
        subprocess.run(["java", "-cp", MV2H_BIN, "mv2h.tools.Converter",  "-i", midi_gt, "-o", "gt.txt",])
        subprocess.run(["java", "-cp", MV2H_BIN, "mv2h.tools.Converter",  "-i", midi_pred, "-o", "pred.txt",])
        subprocess.run(["java", "-cp", MV2H_BIN, "mv2h.tools.Converter",  "-i", midi_gt, "-o", midi_gt + ".conv.txt",])
        subprocess.run(["java", "-cp", MV2H_BIN, "mv2h.tools.Converter",  "-i", midi_pred, "-o", midi_pred + ".conv.txt",])
        result = subprocess.check_output(["java", "-cp", MV2H_BIN, "mv2h.Main", "-g", midi_gt + ".conv.txt", "-t", midi_pred + ".conv.txt"], universal_newlines=True)
    finally:
        try:
            os.remove(midi_gt + ".conv.txt")
            os.remove(midi_pred + ".conv.txt")
        except:
            pass
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
        return quantize(note_seq, annotations, "test.mid")

    def quantize(note_seq, annotations, save_path):
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
        midi = pretty_midi.PrettyMIDI(resolution=480)
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

        for i, (pitch, onset, length, velocity) in enumerate(note_seq):
            # Insert the quantized note into the PrettyMIDI object
            note = pretty_midi.Note(
                velocity=int(velocity),
                pitch=int(pitch),
                start=onset,
                end=onset + length
            )
            if hand_parts is None or hand_parts[i] == 0:
                instrument_left.notes.append(note)
            else:
                instrument_right.notes.append(note)

        midi.instruments.append(instrument_left)
        midi.instruments.append(instrument_right)

        # Tempo/Quantization -----------------------------------------------------
        # Next, we take care of tempo to quantize correctly.
        # We will step from beat to beat, inserting tempo events such that the qpm is
        # correct, yielding exactly "one beat worth of ticks per beat".
        for prev_beat, next_beat in zip(beats, beats[1:]):
            # Calculate the tempo in beats per minute
            # In a compound meters, one inter-beat-interval does not correspond to one
            # quarter note. Therefore we have to rescale according to the time signature
            current_time_signature = midi.time_signature_changes[0]
            for time_signature in midi.time_signature_changes:
                if time_signature.time <= prev_beat:
                    current_time_signature = time_signature
                else:
                    break
            # Calculate seconds per tick
            # Naive:
            # bpm = 60 / inter_beat_interval
            # qpm = bpm / bpm_per_qpm
            # tick_scale = 60 / (qpm * midi.resolution)
            # Simplified:
            bpm_per_qpm = pretty_midi.utilities.qpm_to_bpm(1.0,
                                                           current_time_signature.numerator,
                                                           current_time_signature.denominator)

            tick_scale = (next_beat - prev_beat) * bpm_per_qpm / midi.resolution

            # Insert the tempo change into the PrettyMIDI object
            insert_ticks = midi.time_to_tick(prev_beat)
            midi._tick_scales.append((insert_ticks, tick_scale))
            midi._update_tick_to_time(insert_ticks + 1)

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
        # Load ground truth
        note_sequence, annotations = dataset._load_data(row)
        midi_pred = predict(note_sequence)
        annotations["onsets_musical"] = np.round(annotations['onsets_musical'] * N_per_beat) / N_per_beat
        annotations["onsets_musical"][annotations["onsets_musical"] == N_per_beat] = 0 # reset one beat onset to 0
        # note values
        annotations['note_value'] = np.round(annotations['note_value'] * N_per_beat) / N_per_beat
        annotations['note_value'][annotations['note_value'] == 0] = 1 / N_per_beat

        midi_gt = quantize(note_sequence, annotations, "test_gt.mid")
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
