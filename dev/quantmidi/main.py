import warnings
warnings.filterwarnings('ignore')
import os, sys
sys.path.insert(0, '..')
import argparse
import pytorch_lightning as pl
pl.seed_everything(42)
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import mir_eval
import pretty_midi as pm
from pathlib import Path
from subprocess import check_output
from functools import cmp_to_key, reduce

from quantmidi.data.data_utils import DataUtils
from quantmidi.data.data_module import QuantMIDIDataModule
from quantmidi.models.note_sequence import NoteSequenceModel
from quantmidi.models.baseline import BaselineModel
from quantmidi.models.proposed import ProposedModel
from quantmidi.post_processing import post_process, generate_MIDI_score #, DBN_beat_track
from quantmidi.data.constants import (
    tsIndex2Nume,
    tsIndex2Deno,
    keyNumber2Name,
    N_per_beat,
)

## -------------------------
## DEBUGGING BLOCK
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
torch.autograd.set_detect_anomaly(True)
## END DEBUGGING BLOCK
## -------------------------


def train_or_test(args):

    # ========= get workspace =========
    feature_folder = str(Path(args.workspace, 'features'))
    tracking_uri = str(Path(args.workspace, 'mlruns'))

    # ========= create dataset, model, logger =========
    data_aug_args = {
        'tempo_change_prob': args.tempo_change_prob,
        'tempo_change_range': args.tempo_change_range,
        'pitch_shift_prob': args.pitch_shift_prob,
        'pitch_shift_range': args.pitch_shift_range,
        'extra_note_prob': args.extra_note_prob,
        'missing_note_prob': args.missing_note_prob,
    }
    datamodule = QuantMIDIDataModule(
        feature_folder=feature_folder,
        model_type=args.model_type,
        data_aug_args=data_aug_args,
        workers=args.workers,
        proposed_model_version=args.proposed_model_version,
    )

    if args.model_type == 'note_sequence':
        model = NoteSequenceModel(
            features=args.features,
            pitch_encoding=args.pitch_encoding,
            onset_encoding=args.onset_encoding,
            duration_encoding=args.duration_encoding,
            downbeats=args.downbeats,
            tempos=args.tempos,
            reverse_link=args.reverse_link,
        )
    elif args.model_type == 'baseline':
        model = BaselineModel()
    elif args.model_type == 'proposed':
        model = ProposedModel(version=args.proposed_model_version)

    logger = pl.loggers.MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=tracking_uri,
        run_name=args.run_name,
        tags={
            'option': args.option,
            'model_type': args.model_type,
            'features': ','.join(args.features),
            'pitch_encoding': args.pitch_encoding,
            'onset_encoding': args.onset_encoding,
            'duration_encoding': args.duration_encoding,
            'tempo_change_prob': args.tempo_change_prob,
            'tempo_change_range': ','.join(map(str, args.tempo_change_range)),
            'pitch_shift_prob': args.pitch_shift_prob,
            'pitch_shift_range': ','.join(map(str, args.pitch_shift_range)),
            'extra_note_prob': args.extra_note_prob,
            'missing_note_prob': args.missing_note_prob,
            'downbeats': args.downbeats,
            'tempos': args.tempos,
            'reverse_link': args.reverse_link,
            'proposed_model_version': args.proposed_model_version,
            'workers': args.workers,
            'gpus': args.gpus,
        },
    )

    # ========= create trainer =========
    trainer = pl.Trainer(
        default_root_dir=tracking_uri,
        logger=logger,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        # gpus=args.gpus,
    )

    # ========= train/test =========
    if args.option == 'train':
        if args.resume_training:
            trainer.fit(model, datamodule=datamodule, ckpt_path=args.model_checkpoint)
        else:
            trainer.fit(model, datamodule=datamodule)
    elif args.option == 'test':
        trainer.test(model, datamodule=datamodule, ckpt_path=args.model_checkpoint)

def evaluate(args):

    feature_folder = str(Path(args.workspace, 'features'))
    device = torch.device('cuda') if args.gpus else torch.device('cpu')

    # ========= load pre-trained model from checkpoint =========
    if args.model_type == 'note_sequence':
        model = NoteSequenceModel.load_from_checkpoint(
            args.model_checkpoint,
            features=args.features,
            pitch_encoding=args.pitch_encoding,
            onset_encoding=args.onset_encoding,
            duration_encoding=args.duration_encoding,
            downbeats=args.downbeats,
            tempos=args.tempos,
            reverse_link=args.reverse_link,
        ).to(device)
    elif args.model_type == 'baseline':
        model = BaselineModel.load_from_checkpoint(args.model_checkpoint).to(device)
    elif args.model_type == 'proposed':
        model = ProposedModel.load_from_checkpoint(args.model_checkpoint, version=args.proposed_model_version).to(device)
    else:
        raise ValueError('Invalid model type: {}'.format(args.model_type))
    model.eval()

    # ========= get test set metadata =========
    metadata = pd.read_csv(str(Path(feature_folder, 'metadata.csv')))
    metadata = metadata[metadata['split'] == 'test']
    metadata.reset_index(inplace=True)

    # ========= iterate over the test set and evaluate =========
    evals_beats, evals_downbeats = [], []  # beat-level F-measure
    if args.model_type == 'proposed':
        mv2h_results = []

    for i, row in metadata.iterrows():
        print('Evaluating {}/{}'.format(i+1, len(metadata)))
        # get model input and ground truth annotations
        note_sequence, annotations = pickle.load(open(str(Path(feature_folder, row['feature_file'])), 'rb'))
        note_sequence = note_sequence.unsqueeze(0).to(device)
        beats_targ = annotations['beats']
        downbeats_targ = annotations['downbeats']

        # get model predictions
        if args.model_type == 'baseline':
            length = torch.Tensor([len(note_sequence)]).unsqueeze(0).to(device)
            y_b, y_db = model(note_sequence, length)
            y_b = y_b.squeeze(0).cpu().detach().numpy()
            y_db = y_db.squeeze(0).cpu().detach().numpy()

            beats_pred, downbeats_pred = DBN_beat_track(y_b, y_db)

        elif args.model_type == 'proposed':
            y_b, y_db, y_tempo, y_time_nume, y_time_deno, y_key, y_onset, y_value, y_hands = model(note_sequence)
            y_b = y_b.squeeze(0).cpu().detach().numpy()
            y_db = y_db.squeeze(0).cpu().detach().numpy()
            y_time_nume = y_time_nume.squeeze(0).topk(1, dim=0)[1][0].cpu().detach().numpy()
            y_time_deno = y_time_deno.squeeze(0).topk(1, dim=0)[1][0].cpu().detach().numpy()
            y_key = y_key.squeeze(0).topk(1, dim=0)[1][0].cpu().detach().numpy()
            y_onset = y_onset.squeeze(0).topk(1, dim=0)[1][0].cpu().detach().numpy()
            y_value = y_value.squeeze(0).topk(1, dim=0)[1][0].cpu().detach().numpy()
            y_hands = y_hands.squeeze(0).cpu().detach().numpy()

            # get beats and downbeats predictions
            onsets = note_sequence[0,:,1].cpu().detach().numpy()
            beats_pred, downbeats_pred = post_process(onsets, y_b, y_db)

            # generate MIDI score
            Path.mkdir(Path('outputs'), exist_ok=True)
            midi_score_file = str(Path('outputs', row['performance_id']+'_proposed_test.mid'))

            time_numes = [tsIndex2Nume[tn] for tn in y_time_nume]
            time_denos = [tsIndex2Deno[td] for td in y_time_deno]
            keys = [keyNumber2Name[k] for k in y_key]
            # version 1
            onsets_musical = y_onset / N_per_beat
            note_values = y_value / N_per_beat
            # version 2
            # TO ADD
            hands = np.round(y_hands)

            generate_MIDI_score(
                note_sequence=note_sequence.squeeze(0).cpu().detach().numpy(),
                annotations={
                    'beats': beats_pred,
                    'downbeats': downbeats_pred,
                    'time_signatures': (time_numes, time_denos),
                    'keys': keys,
                    'onsets_musical': onsets_musical,
                    'note_values': note_values,
                    'hands': hands,
                },
                output_file=midi_score_file,
            )
            # import pretty_midi as pm
            # midi_targ = pm.PrettyMIDI(row['midi_perfm'])
            # midi_pred = pm.PrettyMIDI(midi_score_file)
            # notes_targ = midi_targ.instruments[0].notes
            # notes_pred = midi_pred.instruments[0].notes
            # for m in range(30):
            #     print(notes_targ[m], notes_pred[m])
            #     input()
            # print(midi_targ.get_end_time(), midi_pred.get_end_time())
            # input()
            try:
                mv2h_result = mv2h_evaluation(row['midi_perfm'], midi_score_file, args.MV2H_path)
                print(mv2h_result)
                if mv2h_result['Multi-pitch'] > 0.9:
                    mv2h_results.append(mv2h_result)
            except:
                print('pass')

        elif args.model_type == 'note_sequence':
            y_b, y_db = model(note_sequence)
            y_b = y_b.squeeze(0).cpu().detach().numpy()
            y_db = y_db.squeeze(0).cpu().detach().numpy()

            onsets = note_sequence[0,:,1].cpu().detach().numpy()
            beats_pred, downbeats_pred = post_process(onsets, y_b, y_db)

        if args.plot_results:
            segment_length = 30.0
            for seg in range(int(beats_targ[-1] / segment_length)+1):
                l, r = seg * segment_length, (seg+1) * segment_length

                beats_pred_segment = beats_pred[np.logical_and(beats_pred >= l, beats_pred <= r)]
                beats_targ_segment = beats_targ[np.logical_and(beats_targ >= l, beats_targ <= r)]
                downbeats_pred_segment = downbeats_pred[np.logical_and(downbeats_pred >= l, downbeats_pred <= r)]
                downbeats_targ_segment = downbeats_targ[np.logical_and(downbeats_targ >= l, downbeats_targ <= r)]

                plt.figure(figsize=(20,10))
                plt.vlines(beats_pred_segment, 0, 1, color='b', linestyle='--')
                plt.vlines(beats_targ_segment, 1, 2, color='r', linestyle='--')
                plt.vlines(downbeats_targ_segment, 2, 3, color='r', linestyle='--')
                plt.vlines(downbeats_pred_segment, 3, 4, color='b', linestyle='--')
                plt.title('Beat/downbeat tracking results, test piece {}, segment {}'.format(row['performance_id'], seg))
                plt.savefig('debug.png')
                input('enter to continue')

        # evaluate using beat-level F-measure
        f_beats = mir_eval.beat.f_measure(mir_eval.beat.trim_beats(beats_targ), mir_eval.beat.trim_beats(beats_pred))
        f_downbeats = mir_eval.beat.f_measure(mir_eval.beat.trim_beats(downbeats_targ), mir_eval.beat.trim_beats(downbeats_pred))
        evals_beats.append(f_beats)
        evals_downbeats.append(f_downbeats)

    print('\n ======== Beat-level F-measure =========')
    print('Beat tracking:')
    print('F-measure: {:.4f}'.format(np.mean(evals_beats)))
    print('Downbeat tracking:')
    print('F-measure: {:.4f}'.format(np.mean(evals_downbeats)))

    if args.model_type == 'proposed':
        print('\n ======== MV2H evaluation =========')
        print('Multi-pitch: {:.4f}'.format(np.mean([r['Multi-pitch'] for r in mv2h_results])))
        print('Voice: {:.4f}'.format(np.mean([r['Voice'] for r in mv2h_results])))
        print('Meter: {:.4f}'.format(np.mean([r['Meter'] for r in mv2h_results])))
        print('Value: {:.4f}'.format(np.mean([r['Value'] for r in mv2h_results])))
        print('Harmony: {:.4f}'.format(np.mean([r['Harmony'] for r in mv2h_results])))
        print('Average: {:.4f}'.format(np.mean([np.mean([r['Voice'], r['Meter'], r['Value'], r['Harmony']]) for r in mv2h_results])))
        print('MV2H: {:.4f}'.format(np.mean([r['MV2H'] for r in mv2h_results])))


def evaluate_mv2h(args):

    feature_folder = str(Path(args.workspace, 'features'))

    # ========= get test set metadata =========
    metadata = pd.read_csv(str(Path(feature_folder, 'metadata.csv')))
    metadata = metadata[metadata['split'] == 'test']
    metadata.reset_index(inplace=True)

    mv2h_results = []

    for i, row in metadata.iterrows():
        print('Evaluating {}/{}'.format(i+1, len(metadata)))

        midi_targ_file = row['midi_perfm']
        midi_pred_file = str(Path('outputs', row['performance_id']+'_proposed_test.mid'))


        # import pretty_midi as pm
        # import time
        # midi_targ = pm.PrettyMIDI(midi_targ_file)
        # midi_pred = pm.PrettyMIDI(midi_pred_file)
        # # notes_targ = midi_targ.instruments[0].notes
        # # notes_pred = midi_pred.instruments[0].notes
        # # for m in range(30):
        # #     print(notes_targ[m], notes_pred[m])
        # #     input()
        # print('target hand parts:', len(midi_targ.instruments))
        # print(midi_targ.get_end_time(), midi_pred.get_end_time())
        try:
            mv2h_result = mv2h_evaluation(midi_targ_file, midi_pred_file, args.MV2H_path)
            print(mv2h_result)
            # if mv2h_result['Multi-pitch'] > 0.9:
            #     mv2h_results.append(mv2h_result)
            mv2h_results.append(mv2h_result)
        except:
            print('pass')

    print('\n ======== MV2H evaluation =========')
    print('Multi-pitch: {:.4f}'.format(np.mean([r['Multi-pitch'] for r in mv2h_results])))
    print('Voice: {:.4f}'.format(np.mean([r['Voice'] for r in mv2h_results])))
    print('Meter: {:.4f}'.format(np.mean([r['Meter'] for r in mv2h_results])))
    print('Value: {:.4f}'.format(np.mean([r['Value'] for r in mv2h_results])))
    print('Harmony: {:.4f}'.format(np.mean([r['Harmony'] for r in mv2h_results])))
    print('Average: {:.4f}'.format(np.mean([np.mean([r['Voice'], r['Meter'], r['Value'], r['Harmony']]) for r in mv2h_results])))
    print('MV2H: {:.4f}'.format(np.mean([r['MV2H'] for r in mv2h_results])))



def get_test_performance_midis(args):

    feature_folder = str(Path(args.workspace, 'features'))

    # ========= get test set metadata =========
    metadata = pd.read_csv(str(Path(feature_folder, 'metadata.csv')))
    metadata = metadata[metadata['split'] == 'test']
    metadata.reset_index(inplace=True)

    for i, row in metadata.iterrows():
        print('{}/{}'.format(i+1, len(metadata)))

        midi_data_score = pm.PrettyMIDI(row['midi_perfm'])
        notes = reduce(lambda x, y: x+y, [inst.notes for inst in midi_data_score.instruments])
        notes = sorted(notes, key=cmp_to_key(DataUtils.compare_note_order))

        midi_data_perfm = pm.PrettyMIDI()
        midi_data_perfm.instruments.append(pm.Instrument(0))
        for note in notes:
            midi_data_perfm.instruments[0].notes.append(
                pm.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
            )

        Path.mkdir(Path('./outputs_perfm'), parents=True, exist_ok=True)
        midi_perfm_file = str(Path('outputs_perfm', row['performance_id']+'_perfm.mid'))
        midi_data_perfm.write(midi_perfm_file)


def mv2h_evaluation(target_midi_file, output_midi_file, MV2H_path, timeout=20.):
        try:
            output = check_output(['sh', 'evaluate_midi_mv2h.sh',
                                    target_midi_file, output_midi_file, MV2H_path],
                                    timeout=timeout)
        except ValueError as e:
            print('Failed to evaluate pair: \ntarget midi: {}\noutput midi: {}'.format(target_midi_file,
                                                                            output_midi_file))

        # extract result from output
        result_list = output.decode('utf-8').splitlines()[-6:]
        result = dict([tuple(item.split(': ')) for item in result_list])
        for key, value in result.items():
            result[key] = float(value)

        return result

if __name__ == '__main__':

    # ========= parse arguments =========
    parser = argparse.ArgumentParser(description='Main programme for model training.')

    # dataset path and workspace
    parser.add_argument('--dataset_folder', type=str, nargs='+', help='Path to the dataset folders \
                        in the order of ASAP, A_MAPS, CPM, ACPAS')
    parser.add_argument('--workspace', type=str, help='Path to the workspace')
    parser.add_argument('--MV2H_path', type=str, help='Path to the MV2H folder')

    # experiment settings
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment', default='full-training')
    parser.add_argument('--run_name', type=str, help='Name of the run', default='run-0')

    parser.add_argument('--option', type=str, help='Options for the experiment, select from [train, test, \
                        evaluate, evaluate_mv2h]', default='train')
    parser.add_argument('--model_type', type=str, help='Type of the model, select one from [note_sequence, \
                        baseline, proposed]', default='proposed')
    parser.add_argument('--resume_training', type=int, help='Whether to resume training from the last \
                        checkpoint', default=0)
    parser.add_argument('--plot_results', type=int, help='Whether to plot results during evaluation', default=0)

    # input data comparison (features and encoding)
    parser.add_argument('--features', type=str, nargs='+', help='List of features to be used, select one or \
                        more from [pitch, onset, duration, velocity]', default=['pitch', 'onset', 'duration', \
                        'velocity'])
    parser.add_argument('--pitch_encoding', type=str, help='Pitch encoding, select from [midi, chroma]', \
                        default='midi')
    parser.add_argument('--onset_encoding', type=str, help='Encoding of onset features. Select one from \
                        [absolute-raw, shift-raw, absolute-onehot, shift-onehot].', default='shift-raw')
    parser.add_argument('--duration_encoding', type=str, help='Encoding of duration features. Select one \
                        from [raw, onehot].', default='raw')

    # data augmentation parameters
    parser.add_argument('--tempo_change_prob', type=float, help='Probability of tempo change', default=1.0)
    parser.add_argument('--tempo_change_range', type=float, nargs='+', help='Range of tempo change', \
                        default=[0.8, 1.2])
    parser.add_argument('--pitch_shift_prob', type=float, help='Probability of pitch shift', default=1.0)
    parser.add_argument('--pitch_shift_range', type=float, nargs='+', help='Range of pitch shift', \
                        default=[-12, 12])
    parser.add_argument('--extra_note_prob', type=float, help='Probability of extra note', default=0.5)
    parser.add_argument('--missing_note_prob', type=float, help='Probability of missing note', default=0.5)

    # options in note sequence model
    parser.add_argument('--downbeats', type=int, help='Whether to output downbeats or not', default=0)
    parser.add_argument('--tempos', type=int, help='Whether to output tempos or not', default=0)
    parser.add_argument('--reverse_link', type=int, help='Whether to reverse link between beats and \
                        downbeats or not', default=0)

    # proposed model version
    parser.add_argument('--proposed_model_version', type=int, help='Version of the proposed model', default=1)

    # parallelization
    parser.add_argument('--workers', type=int, help='Number of workers for parallel processing', default=8)
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use', default=4)

    parser.add_argument('--model_checkpoint', type=str, help='Path to the checkpoint to resume training \
                        from or for testing/evaluation', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')

    args = parser.parse_args()

    # ========= input check =========
    # dataset_folder
    # workspace
    # experiment_name
    # option
    # model_type
    if args.model_type not in ['note_sequence', 'baseline', 'proposed']:
        raise ValueError('Invalid model type: {}'.format(args.model_type))
    # features
    if args.model_type in ['baseline', 'proposed']:
        if args.features != ['pitch', 'onset', 'duration', 'velocity']:
            print("INFO: Invalid features for {} model. Using default features instead.".format(args.model_type))
            args.features = ['pitch', 'onset', 'duration', 'velocity']
    # workers
    if args.option == 'test' or args.option == 'evaluate':
        if args.workers >= 1:
            print('INFO: Reset number of workers to 0 for testing/evaluation.')
            args.workers = 0
    # gpus
    if args.option == 'test' or args.option == 'evaluate':
        if args.gpus > 1:
            print('INFO: Reset number of GPUs to 1 for testing/evaluation.')
            args.gpus = 1
    # verbose


    # ========= run train/test or evaluate =========
    if args.option in ['train', 'test']:
        train_or_test(args)
    elif args.option == 'evaluate':
        evaluate(args)
    elif args.option == 'evaluate_mv2h':
        evaluate_mv2h(args)
    elif args.option == 'get_test_performance_midis':
        get_test_performance_midis(args)
