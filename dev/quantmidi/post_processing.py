# import madmom
import numpy as np
from enum import IntEnum, auto
from collections import defaultdict
from functools import cmp_to_key
import mido

from quantmidi.data.constants import resolution, tolerance

min_bpm = 40
max_bpm = 220
ticks_per_beat = 240

# # ==================== DBN beat tracker ====================

# transition_lambda = 100.0
# beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(
#     min_bpm=min_bpm,
#     max_bpm=max_bpm,
#     fps=int(1 / resolution),
#     transition_lambda=transition_lambda,
# )
# downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
#     beats_per_bar=[3, 4],
#     min_bpm=min_bpm,
#     max_bpm=max_bpm,
#     fps=int(1 / resolution),
#     transition_lambda=transition_lambda,
# )

# def DBN_beat_track(beat_act, downbeat_act):
#     """
#     Beat tracking using the DBN algorithm.

#     Args:
#         beat_act: beat activation tensor
#         downbeat_act: downbeat activation tensor
#     Returns:
#         beats: beat times
#         downbeats: downbeat times
#     """
#     beats = beat_tracker(beat_act)
#     combined_act = np.vstack((np.maximum(beat_act - downbeat_act, 0), downbeat_act)).T
#     downbeats = downbeat_tracker(combined_act)
#     downbeats = downbeats[:, 0][downbeats[:, 1] == 1]
#     return beats, downbeats

# ==================== Proposed beat tracker ====================


def post_process(
    onsets,
    beat_probs,
    downbeat_probs,
    dynamic_thresholding=True,
    merge_downbeats=True,
    dynamic_programming=True,
):
    """
    Post-processing of beat and downbeat tracking results for proposed model.

    Args:
        onsets: onsets for each note
        beat_probs: beat probabilities for each note
        downbeat_probs: downbeat probabilities for each note
        dynamic_thresholding: whether to use dynamic thresholding or not
    Returns:
        beats: beat times
        downbeats: downbeat times
    """
    N_notes = len(onsets)

    # ========= Dynamic thresholding =========
    if dynamic_thresholding:
        # window length in seconds
        wlen_beats = (60. / min_bpm) * 4
        wlen_downbeats = (60. / min_bpm) * 8

        # initialize beat and downbeat thresholds
        thresh_beats = np.ones(N_notes) * 0.5
        thresh_downbeats = np.ones(N_notes) * 0.5

        l_b, r_b, l_db, r_db = 0, 0, 0, 0  # sliding window indices

        for i, onset in enumerate(onsets):
            # udpate pointers
            while onsets[l_b] < onset - wlen_beats / 2:
                l_b += 1
            while r_b < N_notes and onsets[r_b] < onset + wlen_beats / 2:
                r_b += 1
            while onsets[l_db] < onset - wlen_downbeats / 2:
                l_db += 1
            while r_db < N_notes and onsets[r_db] < onset + wlen_downbeats / 2:
                r_db += 1
            # update beat and downbeat thresholds
            thresh_beats[i] = np.max(beat_probs[l_b:r_b]) * 0.5
            thresh_downbeats[i] = np.max(downbeat_probs[l_db:r_db]) * 0.5

        # threshold beat and downbeat probabilities
        beats = onsets[beat_probs > thresh_beats]
        downbeats = onsets[downbeat_probs > thresh_downbeats]

    else:
        beats = onsets[beat_probs > 0.5]
        downbeats = onsets[downbeat_probs > 0.5]

    # ========= remove beats that are too close to each other =========
    beats_min = beats[np.concatenate([[True], np.abs(np.diff(beats)) > tolerance * 2])]
    beats_max = beats[::-1][np.concatenate([[True], np.abs(np.diff(beats[::-1])) > tolerance * 2])][::-1]
    beats = np.mean([beats_min, beats_max], axis=0)
    downbeats_min = downbeats[np.concatenate([[True], np.abs(np.diff(downbeats)) > tolerance * 2])]
    downbeats_max = downbeats[::-1][np.concatenate([[True], np.abs(np.diff(downbeats[::-1])) > tolerance * 2])][::-1]
    downbeats = np.mean([downbeats_min, downbeats_max], axis=0)

    # ========= merge downbeats to beats if they are not in beat prediction =========
    if merge_downbeats:
        beats_to_merge = []
        for downbeat in downbeats:
            if np.min(np.abs(beats - downbeat)) > tolerance * 2:
                beats_to_merge.append(downbeat)
        beats = np.concatenate([beats, beats_to_merge])
        beats = np.sort(beats)

    if dynamic_programming:
        beats = run_dynamic_programming(beats)

    return beats, downbeats

# ========= dynamic programming ========================================
# minimize objective function:
#   O = sum(abs(log((t[k] - t[k-1]) / (t[k-1] - t[k-2]))))      (O1)
#       + lam1 * insertions                                     (O2)
#       + lam2 * deletions                                      (O3)
#   t[k] is the kth beat after dynamic programming.
# ======================================================================

def run_dynamic_programming(beats, penalty=1.0):

    # ========= fill up out-of-note beats by inter-beat intervals =========
    # fill up by neighboring beats
    wlen = 5  # window length for getting neighboring inter-beat intervals (+- wlen)
    IBIs = np.diff(beats)
    beats_filled = []

    for i in range(len(beats) - 1):
        beats_filled.append(beats[i])

        # current and neighboring inter-beat intervals
        ibi = IBIs[i]
        ibis_near = IBIs[max(0, i-wlen):min(len(IBIs), i+wlen+1)]
        ibis_near_median = np.median(ibis_near)

        for ratio in [2, 3, 4]:
            if abs(ibi / ibis_near_median - ratio) / ratio < 0.15:
                for x in range(1, ratio):
                    beats_filled.append(beats[i] + x * ibi / ratio)
    beats = np.sort(np.array(beats_filled))

    # ============= insertions and deletions ======================
    beats_dp = [
        [beats[0], beats[1]],     # no insertion
        [beats[0], beats[1]],     # insert one beat
        [beats[0], beats[1]],     # insert two beats
        [beats[0], beats[1]],     # insert three beats
    ]
    obj_dp = [0, 0, 0, 0]

    for i in range(2, len(beats)):
        beats_dp_new = [0] * len(beats_dp)
        obj_dp_new = [0, 0, 0, 0]

        # insert x beats
        for x in range(4):
            ibi = (beats[i] - beats[i-1]) / (x + 1)
            objs = []
            for x_prev in range(4):
                o1 = np.abs(np.log(ibi / (beats_dp[x_prev][-1] - beats_dp[x_prev][-2])))
                o = obj_dp[x_prev] + o1 + penalty * x
                objs.append(o)

            x_prev_best = np.argmin(objs)
            beats_dp_new[x] = beats_dp[x_prev_best] + [beats[i-1] + ibi * k for k in range(1, x+1)] + [beats[i]]
            obj_dp_new[x] = objs[x_prev_best]

        beats_dp = beats_dp_new
        obj_dp = obj_dp_new

    x_best = np.argmin(obj_dp)
    beats = beats_dp[x_best]
    return np.array(beats)

def generate_MIDI_score(note_sequence, annotations, output_file):
    print('Generating MIDI score...')

    # get notes and annotations
    pitches = note_sequence[:,0]
    onsets = note_sequence[:,1]
    durations = note_sequence[:,2]
    velocities = note_sequence[:,3]
    length = len(note_sequence)

    beats = sorted(list(annotations['beats']))
    downbeats = annotations['downbeats']
    time_signatures = annotations['time_signatures']
    keys = annotations['keys']
    onsets_musical = annotations['onsets_musical']
    note_values = annotations['note_values']
    hands = annotations['hands']

    if onsets[0] < beats[0]:
        beats = [beats[0] - (beats[1] - beats[0])] + beats
        if beats[0] < 0:
            beats[0] = 0
    while max(onsets+durations) > beats[-1]:
        beats.append(beats[-1] + (beats[-1] - beats[-2]))
    start_time = beats[0]
    end_time = beats[-1]

    time_signature_changes = [[downbeats[0], time_signatures[0][0], time_signatures[1][0]]]
    for i in range(1, length):
        if time_signatures[0][i] != time_signatures[0][i-1] or time_signatures[1][i] != time_signatures[1][i-1]:
            time_signature_changes.append([onsets[i], time_signatures[0][i], time_signatures[1][i]])

    key_changes = [[downbeats[0], keys[0]]]
    for i in range(1, length):
        if keys[i] != keys[i-1]:
            key_changes.append([onsets[i], keys[i]])

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
    # # time signatures
    # for ts in time_signature_changes:
    #     track_0.append(
    #         mido.MetaMessage('time_signature',
    #             time=time2tick(ts[0]),
    #             numerator=ts[1],
    #             denominator=ts[2],
    #         )
    #     )
    # key signature
    for ks in key_changes:
        track_0.append(
            mido.MetaMessage('key_signature',
                time=time2tick(ks[0]),
                key=ks[1],
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
        subticks_o = int(onsets_musical[ni] * ticks_per_beat)
        subticks_b = time2tick(onsets[ni]) % ticks_per_beat
        beat_idx = time2tick(onsets[ni]) // ticks_per_beat
        if abs(subticks_o - subticks_b) < 10:
            onset_ticks = beat_idx * ticks_per_beat + subticks_o
        else:
            onset_ticks = beat_idx * ticks_per_beat + subticks_b

        # udpate tempo changes by onsets
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
        offset_ticks = time2tick(onsets[ni]+durations[ni])
        durticks_d = int(durations[ni]) * ticks_per_beat % ticks_per_beat
        durticks_b = offset_ticks % ticks_per_beat
        beat_idx = offset_ticks // ticks_per_beat
        if abs(durticks_d - durticks_b) < 10:
            offset_ticks = beat_idx * ticks_per_beat + durticks_d
        if offset_ticks < onset_ticks:
            offset_ticks = onset_ticks + 10

        pitch = int(pitches[ni])
        velocity = int(velocities[ni])
        channel = int(hands[ni])
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

    for track in [track_left, track_right]:
        track.sort(key=cmp_to_key(event_compare))
        track.append(mido.MetaMessage('end_of_track', time=track[-1].time+1))
        mido_data.tracks.append(track)

    # udpate track_0 together with tempo changes
    track_0.sort(key=cmp_to_key(event_compare))
    track_0.append(mido.MetaMessage('end_of_track', time=track_0[-1].time+1))
    mido_data.tracks.insert(0, track_0)

    # ticks from absolute to relative
    for track in mido_data.tracks:
        tick = 0
        for event in track:
            event.time -= tick
            tick += event.time

    mido_data.save(filename=output_file)


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