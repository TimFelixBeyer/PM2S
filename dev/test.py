import argparse
from functools import cmp_to_key
import os
import sys
import uuid
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(sys.path[0], ".."))
sys.path.insert(0, "/rhome/beyer/thesis/")
from song import Song
import subprocess
import mido
import numpy as np
import pretty_midi
from data.dataset_base import BaseDataset


from pm2s.constants import keyName2Number, N_per_beat
from pm2s.features.beat import RNNJointBeatProcessor
from pm2s.features.hand_part import RNNHandPartProcessor
from pm2s.features.key_signature import RNNKeySignatureProcessor
from pm2s.features.quantisation import RNNJointQuantisationProcessor
from pm2s.features.time_signature import RNNTimeSignatureProcessor
from mir_eval.transcription import precision_recall_f1_overlap


MV2H_BIN = "/rhome/beyer/reproduction/MV2H/bin"


def mv2h(midi_pred: str, midi_gt: str) -> dict[str, float]:
    """Calls the java implementation of MV2H, which is equivalent to the following commands:
    $ java -cp MV2H_BIN mv2h.tools.Converter -i midi_gt -o midi_gt.conv.txt
    $ java -cp MV2H_BIN mv2h.tools.Converter -i midi_pred -o midi_pred.conv.txt
    $ java -cp MV2H_BIN mv2h.Main -g midi_gt.conv.txt -t midi_pred.conv.txt
    $ rm midi_gt.conv.txt midi_pred.conv.txt
    """
    with TemporaryDirectory() as tmpdir:
        subprocess.run(["java", "-cp", MV2H_BIN, "mv2h.tools.Converter",  "-i", midi_gt, "-o", tmpdir + "gt.conv.txt",])
        subprocess.run(["java", "-cp", MV2H_BIN, "mv2h.tools.Converter",  "-i", midi_pred, "-o", tmpdir + "pred.conv.txt",])
        result = subprocess.check_output(["java", "-cp", MV2H_BIN, "mv2h.Main", "-g", tmpdir + "gt.conv.txt", "-t", tmpdir + "pred.conv.txt"], universal_newlines=True)

    metrics = {}
    for line in result.splitlines():
        key, value = line.split(": ")
        metrics[key] = float(value)
    return metrics


state_dict_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_model_state_dicts")

processor_beat = RNNJointBeatProcessor(os.path.join(state_dict_path, "beat/RNNJointBeatModel.pth"))
processor_beat._model.eval()
processor_beat._model.to("cuda")
processor_quantisation = RNNJointQuantisationProcessor(
    os.path.join(state_dict_path, "quantisation/RNNJointQuantisationModel.pth"),
    os.path.join(state_dict_path, "beat/RNNJointBeatModel.pth"),
)
processor_quantisation._model.eval()
processor_quantisation._model.to("cuda")
processor_quantisation._model.beat_model.eval()
processor_quantisation._model.beat_model.to("cuda")

processor_hand = RNNHandPartProcessor(os.path.join(state_dict_path, "hand_part/RNNHandPartModel.pth"))
processor_hand._model.eval()
processor_hand._model.to("cuda")

processor_time_sig = RNNTimeSignatureProcessor(
    os.path.join(state_dict_path, "time_signature/RNNTimeSignatureModel.pth")
)
processor_time_sig._model.eval()
processor_time_sig._model.to("cuda")
processor_key_sig = RNNKeySignatureProcessor(
    os.path.join(state_dict_path, "key_signature/RNNKeySignatureModel.pth")
)
processor_key_sig._model.eval()
processor_key_sig._model.to("cuda")

def predict(note_seq: np.ndarray, save_path: str):
    # beats_dedicated = processor_beat.process(note_seq)
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
    # from mir_eval.transcription import onset_precision_recall_f1
    # print(f"P/R/F1: ", onset_precision_recall_f1(np.array([(b, b+0.1) for b in annots_gt["beats"]]), np.array([(b, b+0.1) for b in annotations["beats"]])))
    # Create PrettyMIDI object using the above data
    return quantize(note_seq, annotations, save_path)


def quantize(note_seq, annotations, save_path):
    beats = annotations["beats"]
    hand_parts = annotations["hands"]
    key_signatures = annotations["key_signatures"]
    note_values = annotations["note_value"]
    onset_positions = annotations["onsets_musical"]
    time_signatures = annotations["time_signatures"]
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
    # This is complicated by the fact that MIDI ticks correspond to quarter notes,
    # not beats, which means that we have to rescale differently based on the time
    # signature.

    def filter_beat_correspondences(beat_correspondences, min_tempo=20, max_tempo=480):
        """Filters beat correspondences to remove infeasible ones.

        Args:
            beat_correspondences: A list of (beat, seconds) tuples.
            min_tempo: The minimum allowed tempo in beats per minute. (Default: 20)
            max_tempo: The maximum allowed tempo in beats per minute. (Default: 480)

        Returns:
            A list of filtered (beat, seconds) tuples.
        """
        if min_tempo <= 0:
            raise ValueError("min_tempo must be positive.")
        if min_tempo >= max_tempo:
            raise ValueError("min_tempo must be less than max_tempo.")

        beat_correspondences = sorted(beat_correspondences)
        clean = False
        idx = 0
        while not clean:
            clean = True
            for i in range(idx, len(beat_correspondences) - 1):
                prev_beat, prev_seconds = beat_correspondences[i]
                next_beat, next_seconds = beat_correspondences[i + 1]
                tick_scale = (next_seconds - prev_seconds) / (next_beat - prev_beat + 1e-8)
                if not 60 / max_tempo < tick_scale < 60 / min_tempo:
                    # We prefer to keep beats over note onsets as they are more robust.
                    # Beats are always on integer values, so we avoid popping them.
                    if int(next_beat) == next_beat:
                        beat_correspondences.pop(i)
                    else:
                        beat_correspondences.pop(i + 1)
                    idx = max(0, i - 1)
                    clean = False
                    break
        return beat_correspondences

    def fit_tempo(midi, beat_correspondences):
        """Fit the tempo of the MIDI representation such that the beats occur at the desired times.

        Args:
            beats (list[tuple[float, float]]): List of tuples (beat, seconds).
        """

        # Sort the tempo changes by beat
        beat_correspondences = sorted(beat_correspondences)
        # Feasability checks:
        beats = np.array([t[0] for t in beat_correspondences])
        timestamps = np.array([t[1] for t in beat_correspondences])
        if np.any(np.diff(beats) <= 0):
            raise ValueError("Beats must be unique.")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Beats and desired seconds must be strictly increasing.")

        midi._tick_scales = midi._tick_scales[:1]
        midi._update_tick_to_time(1)
        for (prev_beat, prev_seconds), (next_beat, next_seconds) in zip(beat_correspondences, beat_correspondences[1:]):
            # Calculate the tempo in beats per minute
            # In a compound meters, one inter-beat-interval does not correspond to one
            # quarter note. Therefore we have to rescale according to the time signature
            current_time_signature = midi.time_signature_changes[0]
            for time_signature in midi.time_signature_changes:
                if time_signature.time <= prev_seconds:
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

            tick_scale = (next_seconds - prev_seconds) / (next_beat - prev_beat) * (bpm_per_qpm / midi.resolution)
            if tick_scale > 0.03:
                raise ValueError(f"Tick scale change is too large. {prev_beat} {prev_seconds} -> {next_beat}, {next_seconds}")
            # Insert the tempo change into the PrettyMIDI object
            insert_ticks = midi.time_to_tick(prev_seconds)
            midi._tick_scales.append((insert_ticks, tick_scale))
            midi._update_tick_to_time(insert_ticks + 1)

    beat_correspondences = [(i, b) for i, b in enumerate(beats)]
    # We could add note-based tempo changes but they usually make results worse
    # We fit primarily to beats, but secondarily to the predicted note onsets/durations
    # beats_set = set(range(len(beats)))
    # for note, musical_onset in zip(note_seq, onset_positions):
    #     # Find beat closest to the note
    #     beat_idx = np.argmin(np.abs(beats - note[1]))
    #     beat = beats[beat_idx]

    #     if note[1] < beat:
    #         if beat_idx != 0:
    #             prop_in_beat = (beat - note[1]) / (beat - beats[beat_idx - 1] + 1e-8)
    #             current_beat_location = beat_idx - 1 + prop_in_beat
    #         else:
    #             # TODO: Handle this case
    #             continue
    #     else:
    #         if beat_idx < len(beats) - 1:
    #             prop_in_beat = (note[1] - beat) / (beats[beat_idx + 1] - beat + 1e-8)
    #             current_beat_location = beat_idx + prop_in_beat
    #         else:
    #             continue
    #     if abs(current_beat_location - musical_onset) > 0:
    #         continue
    #     beat_location = musical_onset
    #     if beat_location not in beats_set:
    #         beat_correspondences.append((beat_location, note[1]))
    #         beats_set.add(beat_location)

    beat_correspondences = filter_beat_correspondences(beat_correspondences, 4, 2048)

    fit_tempo(midi, beat_correspondences)
    midi.write(save_path)
    return save_path

def quantize_pm2s(note_sequence, annotations, output_file):

    def event_compare(event1, event2):
        secondary_sort = {
            'set_tempo': lambda e: (1 * 256 * 256),
            'time_signature': lambda e: (2 * 256 * 256),
            'key_signature': lambda e: (3 * 256 * 256),
            'lyrics': lambda e: (4 * 256 * 256),
            'text_events' :lambda e: (5 * 256 * 256),
            'program_change': lambda e: (6 * 256 * 256),
            'pitchwheel': lambda e: ((7 * 256 * 256) + e.pitch),
            'control_change': lambda e: (
                (8 * 256 * 256) + (e.control * 256) + e.value),
            'note_off': lambda e: ((9 * 256 * 256) + (e.note * 256)),
            'note_on': lambda e: (
                (10 * 256 * 256) + (e.note * 256) + e.velocity) if e.velocity > 0
                else ((9 * 256 * 256) + (e.note * 256)),
            'end_of_track': lambda e: (11 * 256 * 256)
        }
        if event1.time == event2.time and event1.type in secondary_sort and event2.type in secondary_sort:
            return secondary_sort[event1.type](event1) - secondary_sort[event2.type](event2)
        return event1.time - event2.time
    ticks_per_beat = 240
    resolution = 0.01
    print('Generating MIDI score...')

    # get notes and annotations
    pitches = note_sequence[:,0]
    onsets = note_sequence[:,1]
    durations = note_sequence[:,2]
    velocities = note_sequence[:,3]
    length = len(note_sequence)

    beats = sorted(list(annotations['beats']))
    # downbeats = annotations['downbeats']
    time_signature_changes = annotations['time_signatures']
    key_changes = [[k[0], int(k[1])] for k in annotations['key_signatures']]
    onsets_musical = annotations['onsets_musical']
    note_values = annotations['note_value']
    hands = annotations['hands']

    if onsets[0] < beats[0]:
        beats = [beats[0] - (beats[1] - beats[0])] + beats
        if beats[0] < 0:
            beats[0] = 0
    while max(onsets+durations) > beats[-1]:
        beats.append(beats[-1] + (beats[-1] - beats[-2]))
    start_time = beats[0]
    end_time = beats[-1]

    # create MIDI file
    mido_data = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # get time2tick from beats, in mido, every beat is divided into ticks
    time2tick_dict = np.zeros(int(round(end_time / resolution)) + 1, dtype=int)
    for i in range(1, len(beats)):
        lt = int(round(beats[i-1] / resolution))
        rt = int(round(beats[i]  / resolution))
        ltick = (i-1) * ticks_per_beat
        rtick = i * ticks_per_beat
        time2tick_dict[lt:rt+1] = np.round(np.linspace(ltick, rtick, rt-lt+1)).astype(int)
    def time2tick(time):
        return time2tick_dict[int(round(time / resolution))]

    # track 0 with time and key information
    track_0 = mido.MidiTrack()
    # time signatures
    for ts in time_signature_changes:
        track_0.append(
            mido.MetaMessage('time_signature',
                time=time2tick(ts[0]),
                numerator=int(ts[1]),
                denominator=int(ts[2]),
            )
        )
    # key signature
    for ks in key_changes:
        key_number_to_mido_key_name = [
        'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
        'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am',
        'Bbm', 'Bm']
        track_0.append(
            mido.MetaMessage('key_signature',
                time=time2tick(ks[0]),
                key=key_number_to_mido_key_name[ks[1]],
            )
        )
    # add hand parts in different tracks
    track_left = mido.MidiTrack()
    track_right = mido.MidiTrack()
    # program number (instrument: piano)
    track_left.append(mido.Message('program_change', time=0, program=0, channel=0))
    track_right.append(mido.Message('program_change', time=0, program=0, channel=1))

    # notes
    prev_onset_seconds = 0  # track onsets for tempo update
    prev_onset_ticks = 0
    for ni in range(length):
        # onset musical from onset prediction or from beat prediction
        subticks_o = int(onsets_musical[ni] * ticks_per_beat) % ticks_per_beat
        subticks_b = time2tick(onsets[ni]) % ticks_per_beat
        beat_idx = time2tick(onsets[ni]) // ticks_per_beat
        if abs(subticks_o - subticks_b) < 10:
            onset_ticks = beat_idx * ticks_per_beat + subticks_o
        else:
            onset_ticks = beat_idx * ticks_per_beat + subticks_b

        # update tempo changes by onsets
        if onset_ticks - prev_onset_ticks >= 10:
            tempo = 1e+6 * (onsets[ni] - prev_onset_seconds) / ((onset_ticks - prev_onset_ticks) / ticks_per_beat)
            track_0.append(
                mido.MetaMessage('set_tempo',
                    time=prev_onset_ticks,
                    tempo=int(tempo),
                )
            )
            prev_onset_seconds = onsets[ni]
            prev_onset_ticks = onset_ticks

        # note value from note value prediction or from beat prediction
        offset_ticks = time2tick(onsets[ni] + durations[ni])
        durticks_d = int(durations[ni]) * ticks_per_beat % ticks_per_beat
        durticks_b = offset_ticks % ticks_per_beat
        beat_idx = offset_ticks // ticks_per_beat
        if abs(durticks_d - durticks_b) < 10:
            offset_ticks = beat_idx * ticks_per_beat + durticks_d
        if offset_ticks < onset_ticks:
            offset_ticks = onset_ticks + 10

        pitch = int(pitches[ni])
        velocity = int(velocities[ni])
        if hands is not None:
            channel = int(hands[ni])
        else:
            channel = 0
        track = track_left if channel == 0 else track_right
        track.append(
            mido.Message('note_on',
                time=time2tick(onsets[ni]),
                note=pitch,
                velocity=velocity,
                channel=channel,
            )
        )
        track.append(
            mido.Message('note_off',
                time=time2tick(onsets[ni]+durations[ni]),
                note=pitch,
                velocity=0,
                channel=channel,
            )
        )
    # update track_0 together with tempo changes
    track_0.sort(key=cmp_to_key(event_compare))
    track_0.append(mido.MetaMessage('end_of_track', time=track_0[-1].time+1))
    mido_data.tracks.append(track_0)

    for track in [track_left, track_right]:
        track.sort(key=cmp_to_key(event_compare))
        track.append(mido.MetaMessage('end_of_track', time=track[-1].time+1))
        mido_data.tracks.append(track)

    # ticks from absolute to relative
    for track in mido_data.tracks:
        tick = 0
        for event in track:
            event.time -= tick
            tick += event.time

    mido_data.save(filename=output_file)
    return output_file

def eval(args):
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

        path_pred = f"/tmp/{uuid.uuid4()}.mid"
        path_gt = f"/tmp/{uuid.uuid4()}.mid"

        # Load inputs
        row = dataset._sample_row(idx)
        # Load ground truth
        note_sequence, annotations = dataset._load_data(row)
        print(np.unique(note_sequence[:, 3]))
        midi_pred = predict(note_sequence, path_pred)
        # # annotations["onsets_musical"] = np.round(annotations['onsets_musical'] * N_per_beat) / N_per_beat
        # # annotations["onsets_musical"][annotations["onsets_musical"] == N_per_beat] = 0 # reset one beat onset to 0
        # # note values
        # # annotations['note_value'] = np.round(annotations['note_value'] * N_per_beat) / N_per_beat
        # # annotations['note_value'][annotations['note_value'] == 0] = 1 / N_per_beat
        # # midi_pred = quantize(note_sequence, annotations, "test_gt.mid")
        midi_gt = row["midi_perfm"]
        os.system(f"cp {midi_gt} {path_gt}")

        # midi = pretty_midi.PrettyMIDI(resolution=480)
        # new_ins = pretty_midi.Instrument(program=0, name="Piano")
        # for note in note_sequence:
        #     new_ins.notes.append(pretty_midi.Note(
        #         velocity=int(note[3]),
        #         pitch=int(note[0]),
        #         start=float(note[1]),
        #         end=float(note[1]) + float(note[2]),
        #     ))
        # midi.instruments.append(new_ins)
        # midi.write(path_pred)
        # midi_pred = path_pred

        # Evaluate
        result = mv2h(midi_pred, path_gt)
        for key, value in result.items():
            metrics[key].append(value)
        # print("raw", result)
        # song = Song.from_midi(midi_pred, type="score")
        # song_gt = Song.from_midi(path_gt, type="score")
        # open("me.txt", "w").write("\n".join(str(n) for n in song.notes_musical))
        # open("ms.txt", "w").write("\n".join(str((n[0], n[1], n[2], n[3])) for n in song_gt.notes_musical))
        # print(sum(len(ins.notes) for ins in song.midi.instruments), sum(len(ins.notes) for ins in song_gt.midi.instruments))
        # print(len(song.notes_musical), len(song_gt.notes_musical))
        # print(len(song.notes_seconds), len(song_gt.notes_seconds))

        # print(song.compare(song_gt))

        # for obj in song.mxl.recurse().getElementsByClass(tempo.TempoIndication):
        #     song.mxl.remove(obj, recurse=True)
        # song.mxl.write("midi", path_pred)
        # for obj in song_gt.mxl.recurse().getElementsByClass(tempo.TempoIndication):
        #     song_gt.mxl.remove(obj, recurse=True)
        # song_gt.mxl.write("midi", path_gt)
        # print("Saved from mxl:", mv2h(path_pred, path_gt))

        os.remove(path_pred)
        os.remove(path_gt)
        # break

    for key, value in metrics.items():
        print(key, np.mean(value), "+/-", np.std(value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model.")

    parser.add_argument(
        "--workspace",
        type=str,
        default="..",
        help="Workspace directory.",
    )

    args = parser.parse_args()

    eval(args)


