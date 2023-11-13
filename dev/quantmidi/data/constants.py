import numpy as np

# ========== data representation related constants ==========
## quantisation resolution
resolution = 0.01  # quantization resolution: 0.01s = 10ms
tolerance = 0.05  # tolerance for beat alignment: 0.05s = 50ms
ibiVocab = int(4 / resolution) + 1  # vocabulary size for ibi: 4s = 4/0.01s + 1, index 0 is ignored during training
ibiVocab_new = int(1.5 / resolution) + 1

# =========== time signature definitions ===========
tsDenominators = [0, 2, 4, 8]  # 0 for others
tsDeno2Index = {0: 0, 2: 1, 4: 2, 8: 3}
tsIndex2Deno = {0: 0, 1: 2, 2: 4, 3: 8}
tsDenoVocabSize = len(tsDenominators)

tsNumerators = [0, 2, 3, 4, 6]  # 0 for others
tsNume2Index = {0: 0, 2: 1, 3: 2, 4: 3, 6: 4}
tsIndex2Nume = {0: 0, 1: 2, 2: 3, 3: 4, 4: 6}
tsNumeVocabSize = len(tsNumerators)

# =========== key signature definitions ==========
# key in sharps in mido
keySharps2Name = {0: 'C', 1: 'G', 2: 'D', 3: 'A', 4: 'E', 5: 'B', 6: 'F#',
                  7: 'C#m', 8: 'G#m', 9: 'D#m', 10: 'Bbm', 11: 'Fm', 12: 'Cm',
                  -11: 'Gm', -10: 'Dm', -9: 'Am', -8: 'Em', -7: 'Bm', -6: 'F#m',
                  -5: 'Db', -4: 'Ab', -3: 'Eb', -2: 'Bb', -1: 'F'}
keyName2Sharps = dict([(name, sharp) for sharp, name in keySharps2Name.items()])
# key in numbers in pretty_midi
keyNumber2Name = [
    'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B',
    'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm',
]
keyName2Number = dict([(name, number) for number, name in enumerate(keyNumber2Name)])
keySharps2Number = dict([(sharp, keyName2Number[keySharps2Name[sharp]]) for sharp in keySharps2Name.keys()])
keyNumber2Sharps = dict([(number, keyName2Sharps[keyNumber2Name[number]]) for number in range(len(keyNumber2Name))])
keyVocabSize = len(keySharps2Name) // 2  # ignore minor keys in key signature prediction!

# =========== onset musical & note value definitions ===========
# proposed model version 1:
N_per_beat = 24  # 24 resolution per beat
max_note_value = 4 * N_per_beat  # 4 beats

# proposed model version 2 (precision: 64th note and sextuplet)
# onsets musical
onsetPositions = [0, 1/16, 2/16, 1/6, 3/16, 4/16, 5/16, 2/6, 6/16, 7/16, 8/16, 9/16, 10/16, 4/6, 11/16, 12/16, 13/16, 5/6, 14/16, 15/16]
def onsetPosition2Index(onset_position):
    """Convert onset position in beats into index in onsetPositions"""
    onset_positions_all = np.array(onsetPositions + [1])
    idx = np.argmin(np.abs(onset_positions_all - onset_position))
    if idx == len(onsetPositions):
        idx = 0
    return idx
Index2onsetPosition = onsetPositions
onsetVocabSize = len(onsetPositions)
# note values
noteValues = [
    0,  # 0 reserved for unknown, ignore during training
    4, 2, 1, 1/2, 1/4, 1/8, 1/16,  # whole, half, 4th, 8th, 16th, 32th, 64th notes
    2/3, 1/3, 1/6, 5/6, # triplets, sextuplets
]
# update with tied notes (including dotted notes)
note_values_tied = []
for i in range(1, len(noteValues)):
    for j in range(i+1, len(noteValues)):
        note_value_tied = noteValues[i] + noteValues[j]
        if np.min(np.abs(np.array(noteValues) - note_value_tied)) > 1/32:
            note_values_tied.append(note_value_tied)
noteValues = noteValues + note_values_tied
def noteValue2Index(note_value):
    """Convert note value in beats into index in noteValues"""
    if note_value > 4.5:
        return 0  # set too long note_values to 0
    note_values_all = np.array(noteValues)
    idx = np.argmin(np.abs(note_values_all - note_value))
    return idx
Index2noteValue = noteValues
noteValueVocabSize = len(noteValues)

# ========= model training related constants =========
batch_size_note_sequence = 32  # batch size for training on note sequence model
batch_size_baseline = 8  # batch size for training on baseline model
batch_size_proposed = 16  # batch size for training on proposed model

max_length_note_sequence = 500  # maximum note sequence length for training note sequence model
max_length_pr = int(30 / resolution)  # maximum pianoroll length for training baseline model