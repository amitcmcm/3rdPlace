import os
from joblib import Parallel, delayed
import json
import pickle
import math
import random
import datetime
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 4)
from IPython.display import display
from sklearn.model_selection import StratifiedGroupKFold
import dataset
import model
from torch.utils.data._utils.collate import default_collate
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import Dataset
from model import WalkModule
from collections import defaultdict
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import shutil
from pytorch_lightning.loggers import MLFlowLogger
import mlflow


def stack_store(yps, fs, idxs):
    _fs = np.concatenate(fs)
    _idxs = torch.cat(idxs).cpu().numpy()
    _yps = torch.cat(yps).cpu().numpy()
    _yps[..., 3] = _yps[..., 3:5].mean(-1)
    _yps = _yps[..., :N_TARGETS]

    assert all([len(e) == len(_yps) for e in [_fs, _idxs, _yps]])
    return _yps, _fs, _idxs


def process_store(_yps, _fs, _idxs, pred_dict, ct_dict, age_dict):
    for i in range(len(_yps)):
        ridxs = np.arange(_idxs[i], _idxs[i] + len(_yps[i]))
        for ig in np.unique(ridxs // STORE_SPLIT):
            k = (_fs[i], ig )
            _ = ridxs // STORE_SPLIT == ig
            aidx = ridxs[_] % STORE_SPLIT
            # print(k, _yps[i].shape, aidx.shape, )
            pred_dict.setdefault(k, np.zeros((STORE_SPLIT, N_TARGETS), dtype = np.float32))
            ct_dict.setdefault(k, np.zeros((STORE_SPLIT, N_TARGETS), dtype = np.float32))
            pred_dict[k][aidx] += _yps[i][_]
            ct_dict[k][aidx] += np.ones_like(_yps[i][_])
            age_dict[k] = 0


def flush_store(pred_dict, ct_dict, age_dict, local_path):
    os.makedirs(local_path, exist_ok=True)  # Ensure the local directory exists

    for k in list(ct_dict.keys()):
        # Check if we meet the condition to process this key
        if ((ct_dict[k] >= 1).mean() >= 1.) or ((ct_dict[k] >= 1).mean() >= 0.1 and age_dict[k] > 10):
            # Calculate the processed data
            x = pred_dict[k] / (ct_dict[k] + 1e-5)
            # Serialize the data with pickle
            file_path = os.path.join(local_path, '{}_{:05d}.pkl'.format(k[0].split('/')[-1], k[1]))

            with open(file_path, 'wb') as f:
                pickle.dump(x.astype(np.float32).round(3), f)

            print(file_path, ct_dict[k].mean(), age_dict[k], x.std())
            # Remove processed entries from dictionaries
            pred_dict.pop(k, None)
            ct_dict.pop(k, None)
            age_dict.pop(k, None)

        elif age_dict[k] > 20:
            # Increment age and delete if it exceeds the threshold
            age_dict[k] = 1 + age_dict.get(k, 0)
            if age_dict[k] > 10:
                pred_dict.pop(k, None)
                ct_dict.pop(k, None)
                age_dict.pop(k, None)
                print('del', k)

def load_obj(k):
    # Load and parse JSON file
    with open(k, 'r') as f:
        r = json.load(f)
    r['m'] = os.path.basename(k).split('.')[0]  # Extract the base name without extension
    r['t'] = datetime.datetime.fromtimestamp(os.path.getmtime(k))  # Add last modified timestamp
    return r


# numpy add two arrays, expanding the smaller one on axis 0
def eadd(a, b):
    if isinstance(a, int): return b.copy();
    if a.shape[0] < b.shape[0]:
        a = np.pad(a, ((0, b.shape[0] - a.shape[0]), (0, 0)), mode='constant')
    elif a.shape[0] > b.shape[0]:
        b = np.pad(b, ((0, a.shape[0] - b.shape[0]), (0, 0)), mode='constant')
    return a + b


def load_obj_2(k):
    # Define local path to the data object
    file_path = os.path.join(k) #todo correct path

    # Check if file exists locally
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found in local directory.")

    # Load and decompress the pickle object
    with open(file_path, 'rb') as f:
        r = pickle.load(f)  # Assuming no compression (zd.decompress removed)

    return r

def logit(x): return 1 / (1 + np.exp(-x))
def rlogit(x): return -np.log(1/(x * 0.9999 + 1e-4/2) - 1)


def download_model_local(m, p, source_folder='source/models', target_folder='code/models', params_folder='code/params'):
    # Ensure target directories exist
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(params_folder, exist_ok=True)
    os.makedirs(source_folder, exist_ok=True)


    # Define source and target paths for the model file
    source_file = os.path.join(source_folder, m + '.pt')
    target_file = os.path.join(target_folder, m + '.pt')

    # Copy model file from source to target folder
    if os.path.exists(source_file):
        with open(source_file, 'rb') as src, open(target_file, 'wb') as dst:
            dst.write(src.read())
    else:
        print(f"Model file {source_file} not found.")

    # Save parameters as JSON in the target parameters folder
    params_file = os.path.join(params_folder, m + '.json')
    with open(params_file, 'w') as f:
        json.dump(p, f)




def prep_metadata(defog_metadata, tdcsfog_metadata, daily_metadata, subjects, full=True,
                  expanded=True):
    m1 = defog_metadata.copy()
    m1.insert(3, 'Test', 0)
    m1 = m1.merge(subjects, on=['Subject', 'Visit', ], how='inner')
    assert len(m1) == len(defog_metadata)

    m2 = tdcsfog_metadata.copy()
    m2 = m2.merge(subjects.drop(columns='Visit'), on=['Subject', ], how='inner')
    assert len(m2) == len(tdcsfog_metadata)

    m3 = daily_metadata.copy()
    m3 = m3.merge(subjects, on=['Subject', 'Visit'], how='inner')
    m3.insert(3, 'Test', 0)
    m3.insert(4, 'Medication', (defog_metadata.Medication == 'on').mean())
    m3.drop(columns=[c for c in m3.columns if 'recording' in c], inplace=True)
    assert len(m3) == len(daily_metadata)

    metadata = pd.concat([m1, m2,  # m3
                          ], axis=0)
    metadata.Medication = 1 * (metadata.Medication == 'on')
    metadata.Sex = 1 * (metadata.Sex == 'M')

    if expanded:
        metadata['num_tests'] = metadata.groupby('Subject').transform(lambda x: x.nunique()).Test
        metadata['max_visit'] = metadata.groupby('Subject').transform(lambda x: x.max()).Visit
        metadata['visit_medications'] = metadata.groupby(['Subject', 'Visit']).transform('nunique').Medication
        metadata['UPDRS_On_vs_Off'] = metadata.UPDRSIII_On - metadata.UPDRSIII_Off
        # add 4 columns to dmetadata

    if full:
        # null fix
        metadata['Uon_null'] = 1 * (metadata.UPDRSIII_On.isnull())
        metadata['Uoff_null'] = 1 * (metadata.UPDRSIII_Off.isnull())
        metadata.UPDRSIII_On = metadata.UPDRSIII_On.fillna(metadata.UPDRSIII_On.mean())
        metadata.UPDRSIII_Off = metadata.UPDRSIII_Off.fillna(metadata.UPDRSIII_Off.mean())
        if expanded:
            metadata.UPDRS_On_vs_Off = metadata.UPDRS_On_vs_Off.fillna(metadata.UPDRS_On_vs_Off.mean())

    metadata.set_index('Id', inplace=True)

    if full:
        metadata['Test_Nonzero'] = 1. * (metadata.Test > 0)
        # for i in range(1):
        #     metadata['Test{}'.format(i)] = metadata.Test == i
        metadata.iloc[:, 1:] = metadata.iloc[:, 1:].astype(np.float32)
        metadata.iloc[:, 1:] = (metadata.iloc[:, 1:] - metadata.iloc[:, 1:].mean(0)) / metadata.iloc[:, 1:].std(0)
        metadata.iloc[:, 1:] = metadata.iloc[:, 1:].clip(-3, 3)

    msubject = metadata.Subject
    metadata.drop(columns='Subject', inplace=True)
    if full:
        metadata = metadata.astype(np.float32)

    m3 = m3.set_index('Id')
    assert m3.shape[1] <= metadata.shape[1];
    i = 0
    while m3.shape[1] < metadata.shape[1]:
        m3.insert(m3.shape[1], 'dummy_{}'.format(i), 0);
        i += 1
    m3.iloc[:, -1] = metadata.iloc[:, -1].min()  # yes, hack, for default_metadata in dataset.py

    return metadata, msubject, m3

def process(f):
    # if exists, return stats
    cache_file = 'cache/' + f.split('.')[0] + '.npy'
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        return load(f).shape, os.path.getsize(cache_file)

    # if not, load array,
    df = pd.read_csv('train/' + f)

    # verify
    assert (df.Time == df.index).all()
    assert all(df.columns == ['Time',
                              'AccV', 'AccML', 'AccAP',
                              'StartHesitation', 'Turn', 'Walking',
                              'Valid', 'Task'][:len(df.columns)])
    assert len(df.columns) in [7, 9]

    v = np.zeros((len(df), 12), dtype=np.float32)
    v[:, 6:8] = 1
    v[:, :df.shape[1] - 1] = df.iloc[:, 1:]

    fid = f.split('/')[-1].split('.')[0]
    mult = 100 if 'tdcs' not in f else 128
    for e in events[events.Id == fid].itertuples():
        v[int(round(e.Init * mult)): int(round(e.Completion * mult)), 8] = 1
        v[int(round(e.Init * mult)): int(round(e.Completion * mult)), 9] = e.Kinetic
        v[int(round(e.Init * mult)): int(round(e.Completion * mult)), 10] = 1 - e.Kinetic
    for e in tasks[tasks.Id == fid].itertuples():
        v[int(round(e.Begin * mult)): int(round(e.End * mult)), 11] = task_dict[e.Task]

    # store as compresssed;
    assert v.dtype == np.float32
    # zc = zstd.ZstdCompressor()
    # compr = zc.compress(pickle.dumps(v))
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, v)
    # with open(cache_file, 'wb') as f:
    #     f.write(compr)

    return v.shape, os.path.getsize(cache_file)  # len(compr)

def clip_ereds(x):
    # x = np.concatenate([x, np.zeros((x.shape[0], 1))], 1)
    savg = pd.Series(x[:, -1]).rolling(200, center=True, min_periods=1).mean()
    x[:, -1:] -= max(0.2, np.quantile(savg, 0.15))
    x = x.clip(0, None)  # + 0.01
    return x

def processDaily(f, relabel=False, ):
    cache_file = 'unlabeled/' + 'cache/{}_{:05d}'.format(f, 100) + '.npy'
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        return os.path.getsize(cache_file)
    df = pd.read_parquet(
        os.path.join(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\unlabeled', f),
        columns=['AccV', 'AccML', 'AccAP']).astype(np.float32).round(3)
    # os.remove(DATA_PATH + 'data/' + f + '.zstd')

    # #amit temporary
    # df = pd.read_parquet(r"N:\Projects\ML competition project\winner uploads\5th InnerVoice\local\non-dataset-spec-w-dl\fog@home_preprocessed\07p5n1ucv7.parquet")
    # df = df[['AccV', 'AccML', 'AccAP']].astype(np.float32).round(3)

    assert all(df.columns == ['AccV', 'AccML', 'AccAP', ])
    assert df[::100].std().mean() > 0.1 #amit: why is this needed?
    v = df.values

    maxlen = DAILY_SPLIT
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    klast = None
    plast = None
    for i in range(0, math.ceil(len(v) / maxlen)):
        vsplit = v[i * maxlen:(i + 1) * maxlen] #takes segments of v for example 50000 timepoints
        if relabel:
            k = '{}_{:05d}'.format(unlabeled_files[0].split('/')[-1].split('.')[0], i // 10)
            if k != klast:
                # print(k)
                labels = load_epreds(k)
                labels = clip_ereds(labels)
                labels = cv2.resize(labels, None, fx=1, fy=10)
                klast = k
                plast = labels
            vsave = np.concatenate([vsplit, plast[i % 10 * (maxlen)
                                                  : (i % 10 + 1) * (maxlen)][:len(vsplit)]], 1)
        else:
            vsave = vsplit  # amit
            # print(vsave.shape, vsave.std(0)[3:])
        cache_file = 'cache/unlabeled/{}_{:05d}'.format(f, i) + '.npy'
        # compr = zc.compress(pickle.dumps(vsave))
        np.save(cache_file, vsave.astype(np.float16))

        # with open(cache_file, 'wb') as fc:
        # fc.write(compr)
    return v.shape

def load(f):
    return np.load('cache/' + f.split('.')[0] + '.npy')

def loadDaily(f, i0, i1, verbose=False):
    DAILY_SPLIT = 100000
    vs = []
    # print(f, i0, i1)
    if 'unlabeled/' not in f: f = 'unlabeled/' + f
    if 'cache/' in f: f = f.replace('cache/', '')
    if '.parquet' not in f: f = f + '.parquet'
    # print(f)
    cmin, cmax = i0 // DAILY_SPLIT, (i1 - 1) // DAILY_SPLIT
    for i in range(cmin, cmax + 1):
        cache_file = 'cache/{}_{:05d}'.format(f, i) + '.npy'
        if verbose: print(cache_file)
        if not os.path.exists(cache_file): break;

        with open(cache_file, 'rb') as fc:
            vs.append(
                # pickle.loads(zd.decompress(fc.read()))
                np.load(cache_file)  # .astype(np.float32)
            )
    v = (vs[0] if len(vs) == 1 else np.concatenate(vs)).astype(np.float32)
    return v

class ComboDataset(Dataset):
    ''' combines two datasets, all of the first one
        plus a fraction of the second one, specified as pct of size of first one

        always use random idxs for the second one, so that it's not biased

    '''

    def __init__(self, d1, d2, d2_frac=0.5):
        self.d1 = d1
        self.d2 = d2
        self.d2_frac = d2_frac
        self.d1_len = len(d1)
        self.d2_len = int(self.d1_len * d2_frac)
        self.len = self.d1_len + self.d2_len
        self.d1_len, self.d2_len, self.len
        self.d2_idxs = np.random.choice(len(d2), self.d2_len, replace=False)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx < self.d1_len:
            return self.d1[idx]
        else:
            return self.d2[self.d2_idxs[idx - self.d1_len]]


class CosineMixDataset(Dataset):
    def __init__(self, datasetA, datasetB, n=None):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.n = n or max(len(self.datasetA), len(self.datasetB))
        # self.current_idx = 0
        self.aidxs = np.arange(len(self.datasetA))
        self.bidxs = np.arange(len(self.datasetB))
        np.random.shuffle(self.aidxs)
        np.random.shuffle(self.bidxs)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # self.current_idx += 1
        if random.random() < 0.5 * (1 + np.cos(min(1, idx / self.n) * np.pi)):
            return self.datasetA[self.aidxs[idx % len(self.datasetA)]]
        else:
            return self.datasetB[self.bidxs[idx % len(self.datasetB)]]



if __name__ == '__main__':

    mlflow_logger = MLFlowLogger(experiment_name='exp0', tracking_uri='http://127.0.0.1:7000', save_dir='./mlruns')

    with mlflow.start_run():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(datetime.datetime.now().microsecond)

        np.random.seed(datetime.datetime.now().microsecond)

        os.makedirs("walk", exist_ok=True)

        events = pd.read_csv(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\events.csv')
        subjects = pd.read_csv(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\subjects.csv')
        tasks = pd.read_csv(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\tasks.csv')
        daily_metadata = pd.read_csv(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\daily_metadata.csv')
        defog_metadata = pd.read_csv(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\defog_metadata.csv')
        tdcsfog_metadata = pd.read_csv(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\tdcsfog_metadata.csv')
        sample = pd.read_csv(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\sample_submission.csv')




        all_params = [
            # [{}] * 1,
            # [{'melspec': True, 'n_mels': 16, 'mel_pwr': 0.3}],
            # [{'melspec': True, 'n_mels': 16, 'mel_pwr': None}],

            # [{'seg': True, 'xformer_layers': 0, 'encoder': 'tu-mobilevitv2_100',
            #     'melspec': True, 'n_mels': 16, }],

            [{'xformer_layers': l, 'xformer_init_scale': 0.7, 'rel_pos': e}
             for e in ['mlp', None, ]  # 'bias']
             for l in [3, 4, 5, ]],
            [{'xformer_layers': l, 'deberta': True,
              'xformer_init_1': 1., 'xformer_init_2': 1,
              'xformer_init_scale': 0.7, }
             for l in [3, 4, 5]],
        ]

        def flatten(l): return [item for sublist in l for item in sublist]

        all_params = flatten(all_params)
        all_params = [p.copy() for p in all_params]

        for i, p in enumerate(all_params):
            # p = p.copy()
            p.update({  # 'steps': random.randint(25000, 35000),
                'step_mult': random.choice([2, 3, ]),
                'batch_size': random.choice([12, 16, 20, 24, ]),

                'seq': random.choice([192, 224, 256, 288, 320, 384, ]),
                'patch': random.choice([8, 9, 10, 11, 12, 13]),

                'alibi': random.choice([0, 2, 4, 8]),
                'lion': random.choice([True, False, False]),
                'lr': random.choice([0.3e-4, 0.5e-4, 0.7e-4, 1e-4, 2e-4, 3e-4, 5e-4, ]),
                'weight_decay': round(0.05 * np.exp(np.random.normal(0, 1)), 3),
                'frac_pwr_mult': round(np.exp(np.random.normal(0.7, 0.1)), 2),
                'frac_rand': round(random.random(), 2),
                'stretch_rate': random.choice([0.3, 0.5, 0.7, ]),
                'dims': random.choice([256, ]),
                'act_layer': random.choice(['GELU', 'GELU', 'GELU', 'PReLU', 'CELU']),
                'dropout': random.choice([0.1, 0.15, 0.2, 0.25, 0.3]),
                'focal_alpha': random.choice([0.1, 0.25, 0.25]),
                'focal_gamma': random.choice([1.5, 1.5, 2., 2.5]),
                'patch_act': random.choice(['Identity', 'Identity', 'Identity', 'Identity',
                                            'PReLU', 'PReLU', 'PReLU',
                                            'GELU', 'GELU', 'GELU', 'CELU',
                                            'Tanh', 'Tanh',
                                            'LeakyReLU', 'LeakyReLU', 'LeakyReLU',
                                            ]),
                'rnn': random.choice([None, ] + ['GRU'] * 5 + ['LSTM'] + ['GRU']),
                'se_dims': random.choice([0, 8, 16]),
                'frac_se': False,  # random.choice([True, False]),
                'len_se': False,  # random.choice([False,]),
                'm_se': True,  # random.choice([True, ]),
                'se_dropout': random.choice([0.2, 0.25, 0.3, ]),
                'se_pact': random.choice([0., ]),
                # encodes only defog vs tdcsfog

                'fast_mult': random.choice([1, 1, 1, 1, 0.5, 0.3, ]),
                'final_mult': random.choice([2, 4, 4, 6]),
                'pre_norm': random.choice([True, False]),

                '0x2d57c2': random.choice(['22', '21', '12', ]),
                '0xe86b6e': random.choice(['12', '12', '11']),
                'fix_final': random.choice([True, True, False, ]),
                'mae_divisor': random.choice([1, 2, 5, 10, ]),

                'aux_wt': random.choice([0.]),
                'v_wt': random.choice([0.03, ]),
                'min_wt': random.choice([3e-3, 0.01, ]),

                'frac_adj': random.choice([True, False]),
                'm_adj': random.choice([True, True, True, False]),

                'adj_gn': random.choice([0.3, 0.5, ]),
                'm_adj_gn': random.choice([0.1, 0.2]),

                'len_adj': random.choice([True, False, ]),

                'folds': random.choice(['A', 'B', ]),  # 'C', 'D'])
                'patch_dropout': random.choice([0, 0, 0.,
                                                0.05, 0.1, 0.15, 0.2, 0.25]),
                # 'frac_gn': random.choice([0., 0.03, 0.1]),

                'expanded': random.choice([False, False, False]),
            })

        for p in all_params:
            if p['pre_norm']:
                p['xformer_init_1'] = 1.
                p['xformer_init_2'] = 1.
                p['xformer_init_scale'] = 0.7

            # if random.random() < 1/10 and p['rnn'] is None:
            #     p['xformer_layers'] = 0

            if random.random() < 1:
                p['dims'] = 384
                p['nheads'] = 12
                p['final_mult'] = 4

            if random.random() < 1:
                p['xformer_attn_drop_rate'] = 0.
                p['xformer_drop_path_rate'] = 0.


        RELABEL = False #in his original code = True
        if RELABEL:
            for p in all_params:
                p['relabel'] = True
                p['batch_size'] = 12
                p['steps'] = random.choice([40000, 30000 ])
                p['focal_alpha'] = 0.25
                p['focal_gamma'] = 1.
                p['neg_mult'] = random.choice([0.01, 0.03, 0.1, ])
                if p['seq'] == 192:
                    p['seq'] *= 2


        # was commented in his code, but can't load his eparams
        eparams = random.choice(all_params)
        eparams['seed'] = random.randint(0, 100000)
        eparams['n_folds'] = 4
        eparams['fold'] = 0#random.randint(0, 2)
        eparams['folds'] = 'A'

        params = eparams.copy()
        display(params)

        mlflow.log_params(params)

        for cur_fold in range(4):
            print('FOLD: '+str(cur_fold))

            for directory in ["cache/unlabeled", "cache/tdcsfog", "cache/defog"]:
                if os.path.exists(directory):
                    shutil.rmtree(directory)

            FOLD, SEED = cur_fold, params['seed'] #params['fold'], params['seed']
            N_FOLDS = params.get('n_folds', 3)


            metadata, msubject, dmetadata = prep_metadata(defog_metadata, tdcsfog_metadata, daily_metadata, subjects,
                                               expanded = params['expanded'])

            assert metadata.shape[1] == dmetadata.shape[1]


            downsample = {k[2:]: v for k, v in params.items() if k.startswith('0x')}
            display(downsample)

            drop_ids = []; id_frac = {}
            for k, v in downsample.items():
                print(k, v)
                df = metadata[msubject.loc[metadata.index] == k]
                ids = df.sample(frac = 1 - 1 / int(v[0]), random_state = SEED).index.tolist()
                drop_ids.extend(ids); print(ids)

                ids = {i: 1 / int(v[1]) for i in df.index}
                id_frac.update(ids); print(ids)
                print()


            drop_events = events.Id.isin(drop_ids); print(drop_events.sum())
            event_frac = events.Id.map(id_frac).fillna(1); print((event_frac < 1).sum())

            if params['folds'] in ['A', 'D']:
                # pd cut -- size
                events['Length'] = (events.Completion - events.Init)
                random.seed(SEED)
                events['Q'] = events.groupby('Type')['Length'].transform(
                    lambda x: pd.cut(x, (10 if params['folds'] == 'A'
                                            else random.randint(5, 15)
                                         ) ** np.arange(0, 10), labels = False)).fillna(0)
                assert events.Q.isnull().sum() == 0
                random.seed(datetime.datetime.now().microsecond)
                display(events.groupby(['Type', 'Q']).Length.agg(['mean', 'count', 'sum']))

            elif params['folds'] in ['B', 'C']:
                # pd cut -- binary
                events['Length'] = (events.Completion - events.Init) #* (~drop_events) * event_frac
                random.seed(SEED)
                events['Q'] = events['Length'].transform(
                    lambda x: pd.cut(x, [-1, 5 + 10 * random.random(), 1000000], labels = False) )#.fillna(0)
                assert events.Q.isnull().sum() == 0
                random.seed(datetime.datetime.now().microsecond)
                display(events.groupby(['Type', 'Q']).Length.agg(['mean', 'count', 'sum']))


            ev = sorted(events.Type.dropna().unique())[::]; print(ev)
            f = ~events.Type.isnull() & (events.Type != 'Turn') & (events.Length > 0.5) & ~drop_events
            ef = events[f].set_index('Id')
            etables = []
            s = ef.Type.map(dict(zip(ev, range(1, 1 + len(ev))))) * 100 + 10 * ef.Q
            # something to do with organizing fog events by length (Q=0 is shorter compared to Q=1 etc.), and then the part with ev is something with the proportion of each class.
            # so the resulting value in s is something that takes into account both the proportion of the class and the length of the event.
            # for example, walking events of length Q=0 will all have the value 300 in s. start hesitation with Q=1 will have 110. walking + Q=1 will have 310.
            # the idea is probably to be some sort of method to balance events later.


            # if params['folds'] in ['B', 'C', ]:
            for t, v in zip(ef.itertuples(), s):
                etables.extend([(t.Index, v)] * (
                    int(round(t.Length) ** (0.5 if params['folds'] in ['C', 'D'] else 1))
                            if params['folds'] in ['B', 'C', 'D'] else 1
                            ))

            etable = pd.Series(*list(zip(*etables))[::-1])
            etable = pd.concat((etable, #*([etable[etable % 100 > 0]] * 3), *([etable[etable % 100  > 10]] * 10),
                                pd.Series(0, list(set(metadata.index) - set(etable.index)))))
            etable = (etable + 1 * etable.index.isin(tdcsfog_metadata.Id)).astype(int)
            esubject = msubject.reindex(etable.index)
            etable.value_counts()



            folds = list(StratifiedGroupKFold(n_splits = N_FOLDS,
                                            shuffle = True, random_state = SEED
                    ).split(np.zeros(len(etable)), etable, groups = esubject))
            train_fold, test_fold = folds[FOLD]
            train_ids = etable.iloc[train_fold].index
            test_ids = etable.iloc[test_fold].index
            assert set(msubject.loc[train_ids]) & set(msubject.loc[test_ids]) == set()

            print(len(train_ids), len(test_ids))

            train_subjects = msubject.loc[train_ids].unique()
            test_subjects = msubject.loc[test_ids].unique()
            assert set(train_subjects) & set(test_subjects) == set()
            print(len(train_subjects), len(test_subjects))

            train_df = daily_metadata[daily_metadata.Subject.isin(list(train_subjects))]
            test_df = daily_metadata[~daily_metadata.Subject.isin(list(train_subjects))]

            train_daily_ids, test_daily_ids = train_df.Id.tolist(), test_df.Id.tolist()
            train_daily_subjects, test_daily_subjects = train_df.Subject, test_df.Subject
            assert set(train_daily_ids) & set(test_daily_ids) == set()
            assert set(train_daily_subjects) & set(test_daily_subjects) == set()
            print(len(train_daily_ids), len(test_daily_ids))
            print(len(train_daily_subjects), len(test_daily_subjects))


            s2 = subjects[subjects.Subject.isin(daily_metadata.Subject) & (subjects.NFOGQ == 0)].Subject.unique()
            assert len(set(s2) & set(train_daily_subjects)) == 0


            # fog_files = [f for f in files if 'train/' in f and 'fog' in f]# and not any([z in f for z in common_files])]
            # fog_files = os.listdir(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\train\defog' +
            #                        r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\train\tdcsfog' +
            #                        r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\train\notype')
            fog_files = [f"{subfolder}/{file}"
                         for subfolder in os.listdir(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\train')
                         if subfolder != 'notype' and os.path.isdir(os.path.join(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\train', subfolder))
                         for file in os.listdir(os.path.join(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\train', subfolder))]
            print(len(fog_files))

            # unlabeled_files = [f for f in files if 'unlabeled/' in f]# and not any([z in f for z in common_files])]
            unlabeled_files = os.listdir(r'E:\kaggle dataset\tlvmc-parkinsons-freezing-gait-prediction\unlabeled')
            print(len(unlabeled_files))


            task_dict = dict(zip(sorted(tasks.Task.unique()), np.arange(1, 1 + len(tasks.Task.unique()))))
            print(len(task_dict))

            DAILY_SPLIT = 100000  # 100000 if not RELABEL else 50000

            os.makedirs('cache/unlabeled', exist_ok=True)

            # process cache
            r = Parallel(n_jobs = 3)(delayed(processDaily)(f, RELABEL)
                for f in tqdm(unlabeled_files, total = len(unlabeled_files))
                if any([z in f for z in ( train_daily_ids if RELABEL
                        else random.sample(train_daily_ids, k = min(10, len(train_daily_ids))) )
                                         + test_daily_ids])
                )


            # get the last file (sorted), based on prefix prior to ., for a list
            uc = 'cache/unlabeled/'
            f = sorted([f for f in os.listdir('cache/unlabeled')
                                    # if any([z in f for z in daily_ids])
                            ], )#key = lambda x: x.split('.')[0].split('_')[-1])
            ucount = {}
            for e in f: ucount[uc + e.split('.')[0]] = (int(e.split('_')[-1].split('.')[0]) + 1) * DAILY_SPLIT
            if RELABEL: assert len(ucount) == len(unlabeled_files)
            print(sum(ucount.values()) / 1e6)


            # process all files
            r = Parallel(os.cpu_count())(delayed(process)(f) for f in tqdm(fog_files[:], total=len(fog_files[:])))

            # display counts;
            lcount = dict(zip(fog_files, [e[0][0] for e in r]))
            [sum([v for k, v in lcount.items() if s in k]) / 1e6
                    for s in ['/defog/', '/tdcsfog/', ]]

            PATCH = params.get('patch', 12)
            SEQ = params.get('seq', 256)
            print(PATCH * SEQ / 100)



            train_data = dataset.WalkDataset(
                {k: v for k, v in lcount.items()
                            if k.split('/')[-1].split('.')[0] in train_ids
                            and k.split('/')[-1].split('.')[0] not in drop_ids
                            },
                            metadata, load, loadDaily, -1, id_frac = id_frac,
                            test = False, **dataset.getParams(dataset.WalkDataset, params))
            pure_train_data = train_data

            test_data = dataset.WalkDataset(
                {k: v for k, v in lcount.items()
                            if k.split('/')[-1].split('.')[0] in test_ids},
                              metadata, load, loadDaily, -1,
                              test = True,# id_frac = test_id_frac,
                              **dataset.getParams(dataset.WalkDataset, params))
            train_daily_data = dataset.WalkDataset(
                {k: v for k, v in ucount.items()
                            if k.split('/')[-1] in train_daily_ids},
                            dmetadata if RELABEL else metadata, load, loadDaily, DAILY_SPLIT,
                            test = False,
                            **dataset.getParams(dataset.WalkDataset, params))
            test_daily_data = dataset.WalkDataset(
                {k: v for k, v in ucount.items()
                            if k.split('/')[-1] in test_daily_ids},
                            metadata, load, loadDaily, DAILY_SPLIT,
                            test = True,
                            **dataset.getParams(dataset.WalkDataset, params))

            print(len(train_data))
            print(len(test_data))
            print(len(train_daily_data))
            print(len(test_daily_data))



            if RELABEL:
                train_data = CosineMixDataset(train_daily_data, pure_train_data,
                                                params['batch_size'] * params['steps'])

            else:
                train_data = ComboDataset(train_data, train_daily_data, 0.2)

            print(len(train_data))
            x, y, s, frac = train_data[random.choice(range(len(train_data)))][:4]

            import model
            model = model.WalkModule(params, **model.getParams(WalkModule, params)).to(device)

            print(params)
            model.train()

            x, y, s, frac, m, f, i, flen = default_collate(
                [train_data[random.choice(range(len(train_data)))] for i in range(32)])

            yp = model(x.to(device),
                        m.to(device),
                        frac.to(device),
                        flen.to(device),
                        # adjust = False
                        )[0]
            loss = yp.mean()
            loss.backward()
            print(yp.mean().item(), yp.shape)

            print({k: v for k,v in params.items() if ('wt' in k) and '_' in k})
            print({k: v for k,v in params.items() if ('se' in k or 'frac' in k or 'adj' in k or 'm_' in k) and '_' in k})


            train_loader = DataLoader(train_data, batch_size=params['batch_size'],
                                      shuffle=True if not RELABEL else False,
                                      drop_last=True,
                                      # num_workers=os.cpu_count())
                                      num_workers=0) #amit

            test_loader = DataLoader(test_data, batch_size=32,
                                     num_workers=os.cpu_count())
            test_daily_loader = DataLoader(test_daily_data, batch_size=32,
                                           num_workers=os.cpu_count())

            print((len(train_loader), len(test_loader), len(test_daily_loader)))

            if RELABEL:
                model.params.steps = len(train_loader)
            else:
                model.params.steps = int(params['step_mult'] * (len(train_data) * len(train_loader)) ** 0.5)

            print(model.params.steps) # it looks like (see val_check_interval and max_steps in trainer definition below) he defines the performance of validation (and hence, the times when entering validation_step) to happen after the same number of steps as the total steps. this causes validation to only occur at the end of training.

            # and show progress bar every 10 steps
            trainer = pl.Trainer(accelerator = 'auto', logger = mlflow_logger,
                                 precision = 16,
                                 gradient_clip_val = 1,
                                    enable_checkpointing = False,
                                    val_check_interval = model.params.steps, #when val_check_interval and check_val_every_n_epoch are 1, run is very slow
                                    check_val_every_n_epoch = None,
                                    max_steps = model.params.steps,
                                    callbacks = [pl.callbacks.TQDMProgressBar(refresh_rate = 5)],
                                    )

            trainer.fit(model, train_loader, test_loader)


            N_TARGETS = 4
            STORE_SUBSAMPLE = 10
            STORE_SPLIT = 100000

            # saving results

            m = ''.join(random.choices('0123456789abcdef', k=6))

            try:
                # Define the local directory and ensure it exists
                local_path = os.path.join(r"N:\Projects\ML competition project\Winning Code Local\3rd place\amit", "walk", "preds")
                os.makedirs(local_path, exist_ok=True)

                # Construct the file path
                file_path = os.path.join(local_path, f"{m}.pkl")

                # Save the pickled data locally
                with open(file_path, 'wb') as f:
                    pickle.dump((model.pred_dict, model.target_dict, model.ct_dict), f)

                print(f"Saved to {file_path}")

            except Exception as e:
                print(e)


            try:
                # Define the local directory and ensure it exists
                local_path = os.path.join(r"N:\Projects\ML competition project\Winning Code Local\3rd place\amit", "walk", "models")  # Update to your desired path
                os.makedirs(local_path, exist_ok=True)

                # Construct the file path
                file_path = os.path.join(local_path, f"{m}.pt")

                # Save the model state_dict locally
                with open(file_path, 'wb') as f:
                    pickle.dump(model.model.state_dict(), f)

                print(f"Saved model state_dict to {file_path}")

            except Exception as e:
                print(e)


            # k = random.choice(list(model.pred_dict))
            # plt.plot(model.pred_dict[k] / model.ct_dict[k])
            # plt.plot(model.target_dict[k])
            # plt.ylim(0, 1.03);



            # Prepare the data dictionary `r` with parameters and results
            r = {}
            r['params'] = params
            r['results'] = {k: v.item() if torch.is_tensor(v) else v
                            for k, v in trainer.callback_metrics.items()}

            # Define the local directory and ensure it exists
            local_path = os.path.join(r"N:\Projects\ML competition project\Winning Code Local\3rd place\amit", "walk", "results") # Update to your desired path
            os.makedirs(local_path, exist_ok=True)

            # Construct the file path
            file_path = os.path.join(local_path, f"{m}.json")

            # Save the dictionary as a JSON file locally
            with open(file_path, 'w') as f:
                json.dump(r, f)

            print(f"Saved results to {file_path}", r)


    # loading results

    current_time = datetime.datetime.now()

    # Collect keys (file paths) of JSON files modified in the last 4.3 hours
    keys = [
        os.path.join(local_path, filename)
        for filename in os.listdir(local_path)
        if filename.endswith('.json') and (
                current_time - datetime.datetime.fromtimestamp(
            os.path.getmtime(os.path.join(local_path, filename))) < datetime.timedelta(hours=5) #was 4.3
            # os.path.getmtime(os.path.join(local_path, filename))) #todo amit return to 4.3 hours condition
        )
    ]

    # Load objects in parallel
    max_parallel = min(os.cpu_count() * 3, 60)  # Limit to 60 to stay within the Windows handle limit
    results = Parallel(n_jobs=max_parallel)(delayed(load_obj)(k) for k in keys)


    presults = defaultdict(list)
    ms = {};
    pms = defaultdict(list)
    # f = random.choice(results)['params']['frac_pwr_mult']
    for r in results:
        p = r['params']

        if 'seg' in p and p['seed'] < 400: continue;
        # if p['seed'] //100 in [4, 5]: continue;
        # if p['seq'] < 224: continue;
        # if p['dims'] < 384: continue;
        # if p.get('xformer_layers', 2) > 3: continue;
        # if p['frac_pwr_mult'] != f: continue
        # if p['se_dims'] > 0 and p.get('se_act', '') == 'PReLU': continue;
        r_ = {k: v for k, v in r['params'].items()
              if k not in ['fold', 'seed', 'n_folds']}
        presults[json.dumps(r_)].append(r['results']['val_ap'])
        pms[json.dumps(r_)].append(r['m'])
        ms[r['m']] = r['params']


    # dict to df, each element is row
    df = pd.DataFrame({k: (np.mean(v), np.min(v), np.max(v), len(v))
                       for k, v in presults.items()}, index=['mean', 'min', 'max', 'ct']).T.sort_values('mean')[::-1]
    # df = df.loc[[e for e in df.index
    #              if not any( [z in e for z in [ 'steps', 'h0', 'patch', 'mult', 'seq', 'layers' ]] ) ]]
    # df = df[df.ct >= 4]  # params['n_folds'] == 0] #amit moved to comment to ignore folds
    # df -= df.loc['{}']#max()
    df.round(3)

    select_ms = flatten([pms[e] for e in df.head(10).index
                         # if 'frac_adj": false' in e
                         ])

    df1 = pd.DataFrame([json.loads(k) for k in presults])
    df2 = pd.DataFrame([(np.mean(v), np.min(v), np.max(v), len(v))
                        for k, v in presults.items()], columns=['mean', 'min', 'max', 'ct'])

    pd.set_option('display.max_columns', 200)
    pd.concat((df1, df2), axis=1).sort_values('mean')[::-1][df2.ct >= 3  # df2.ct.max()
                                                            ].round(6).iloc[:, 50:
    # ][[c for c in df1.columns if '_' not  in c]
    ]

    COMMON = ['2d57c2', 'e86b6e'][:2]
    common_files = (tdcsfog_metadata.Id[tdcsfog_metadata.Subject.isin(COMMON)].tolist()
                    + defog_metadata.Id[defog_metadata.Subject.isin(COMMON)].tolist())

    pred_totals = {}
    target_totals = {}
    ct_totals = {}
    t_ct_totals = {}
    scales = []
    for m, p in ms.items():
        try:
            if m not in select_ms: continue
            r = load_obj_2("walk" + '/preds/' + m + '.pkl')
            mult = 1  # /8 if p.get('seg') else 1; #print(mult)

            pred, target, ct = r
            # print(len(pred))
            pred, target, ct = [{k: v for k, v in p.items()
                                 if not any([c in k for c in common_files])}
                                for p in [pred, target, ct]]
            # print(len(pred))
            spred = {k: pred[k] / (ct[k] + 1e-5) for k in pred.keys()}
            pred_total = np.stack([e.sum(0) for e in spred.values()]).sum(0)
            total = np.stack([e.sum(0) for e in target.values()]).sum(0)
            scale = total / pred_total * mult
            scales.append(scale)
            print(scale.round(2))

            # print(m, pred_total, total, scale.round(2)); #break;
            for k, v in spred.items():
                pred_totals[k] = eadd(pred_totals.get(k, 0), v * scale)  #
                # * logit(rlogit(v) + np.log(scale)) )
            for k, v in target.items():
                target_totals[k] = eadd(target_totals.get(k, 0), v)
                t_ct_totals[k] = eadd(t_ct_totals.get(k, 0), 1 * (v > -np.inf))
            for k, v in ct.items():
                ct_totals[k] = eadd(ct_totals.get(k, 0), (v > 0) * mult)
        except Exception as e:
            # raise e
            print('error', m, e)
        # break;
    assert (set(pred_totals.keys()) == set(target_totals.keys())
            == set(ct_totals.keys()) == set(t_ct_totals.keys()))


    # plt.plot(np.stack(scales))

    # # k = random.choice(list(hcommon_files))#.keys()))
    # k = random.choice(list(pred_totals.keys()))
    # plt.plot(pred_totals[k] / ct_totals[k])
    # plt.plot(target_totals[k] / t_ct_totals[k])
    # plt.ylim(0, 1.05);

    keys = [f for f in ct_totals.keys() if not any([z in f for z in common_files])]
    print('{} of {}'.format(len(keys), len(ct_totals)))


    ct_dict, pred_dict, target_dict = ct_totals, pred_totals, target_totals
    final_ys, final_yps = [], []
    for k in keys:
        minlen = min([e.shape[0] for e in [ct_dict[k], pred_dict[k], target_dict[k]]])
        # print(minlen)
        # minlen = (ct_totals[k] > 0).sum()
        # print(minlen)
        # break;
        f = ct_dict[k][:minlen][:, 0] > 0
        final_ys.append(1 * (target_dict[k][:minlen][f] / t_ct_totals[k][:minlen][f] > 0.5))
        final_yps.append(pred_dict[k][:minlen][f] / ct_dict[k][:minlen][f])
        assert (ct_dict[k].std(1) < 1e-5).all()
        assert ct_dict[k][:minlen][f].min() >= 1

    final_ys, final_yps = np.concatenate(final_ys), np.concatenate(final_yps)

    aps = []
    N = 1
    labels = 'htw'
    for i in range(3):
        aps.append(average_precision_score(final_ys[::N, i], final_yps[::N, i]))

    print(aps, np.mean(aps), len(ms))

    A=4


    # # Remove existing directories if they exist # amit in comment for now because i think it's unnecessary and related to creating a kaggle dataset
    # for directory in ["code/models", "code/params"]:
    #     if os.path.exists(directory):
    #         shutil.rmtree(directory)
    #
    # # Recreate the directories
    # for directory in ["code/models", "code/params"]:
    #     os.makedirs(directory, exist_ok=True)
    #
    #
    # # Example parallel execution for models
    # r = Parallel(os.cpu_count() * 3)(delayed(download_model_local)(m, p) for m, p in ms.items())
