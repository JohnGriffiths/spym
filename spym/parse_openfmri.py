import os
import glob

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm

from joblib import Parallel, delayed


def get_scan_key(study_dir):
    """Parse scan_key file to get scanning information (currently only TR).
    """
    with open(os.path.join(study_dir, 'scan_key.txt')) as f:
        scan_key = dict(zip(*np.hsplit(np.array(f.read().split()), 2)))
    scan_key['TR'] = float(scan_key['TR'])
    return scan_key


def get_subjects_id(study_dir):
    return [os.path.split(p)[1]
    for p in glob.glob(os.path.join(study_dir, 'sub*'))
                       if os.path.isdir(p)]


def get_task_sessions(subject_dir):
    """Get the list of scanning sessions with the performed task ids.
    """
    sessions = os.path.join(subject_dir, 'BOLD', '*')
    return [os.path.split(session)[1].split('_')[0]
            for session in sorted(glob.glob(sessions))]


def get_motion(subject_dir):
    """Get the motion parameters for this subject
    """
    sessions = os.path.join(subject_dir, 'BOLD', '*')

    motion = []
    for session_dir in sorted(glob.glob(sessions)):
        motion_s = open(os.path.join(session_dir, 'motion.txt')).read()
        motion_s = np.array([l.split() for l in motion_s.split('\n')][:-1])
        motion.append(np.array(motion_s).astype('float'))
    return motion


def get_scans_count(subject_dir):
    sessions = os.path.join(subject_dir, 'BOLD', '*')

    n_scans = []
    for session_dir in sorted(glob.glob(sessions)):
        img = nb.load(os.path.join(session_dir, 'normalized_bold.nii.gz'))
        n_scans.append(img.shape[-1])

    return n_scans


def get_scans(subject_dir):
    sessions = os.path.join(subject_dir, 'BOLD', '*')

    scans = []
    for session_dir in sorted(glob.glob(sessions)):
        img = nb.load(os.path.join(session_dir, 'normalized_bold.nii.gz'))
        scans.append(img)

    return scans


def get_task_contrasts(study_dir, subject_dir, model_id, hrf_model):
    contrasts_path = os.path.join(
        study_dir, 'models', model_id, 'task_contrasts.txt')

    task_contrasts = {}
    for line in open(contrasts_path, 'rb').read().split('\n')[:-1]:
        line = line.split()
        task_id = line[0]
        contrast_id = line[1]
        con_val = np.array(line[2:]).astype('float')
        if 'with derivative' in hrf_model:
            con_val = np.insert(con_val, np.arange(con_val.size) + 1, 0)
        task_contrasts.setdefault(task_id, {}).setdefault(contrast_id, con_val)

    ordered = {}
    for task_id in sorted(task_contrasts.keys()):
        for contrast_id in task_contrasts[task_id]:
            for session_task_id in get_task_sessions(subject_dir):
                if session_task_id == task_id:
                    con_val = task_contrasts[task_id][contrast_id]
                else:
                    a_con_id = task_contrasts[session_task_id].keys()[0]
                    n_conds = len(task_contrasts[session_task_id][a_con_id])
                    con_val = np.array([0] * n_conds, dtype='float')
                ordered.setdefault(contrast_id, []).append(con_val)
    return ordered


def get_events(subject_dir):
    sessions = os.path.join(subject_dir, 'model', 'model001', 'onsets', '*')

    events = []

    for session_dir in sorted(glob.glob(sessions)):
        conditions = glob.glob(os.path.join(session_dir, '*.txt'))
        onsets = []
        cond_id = []
        for i, path in enumerate(sorted(conditions)):
            cond_onsets = open(path, 'rb').read().split('\n')
            cond_onsets = [l.split() for l in cond_onsets[:-1]]
            cond_onsets = np.array(cond_onsets).astype('float')

            onsets.append(cond_onsets)
            cond_id.append([i] * cond_onsets.shape[0])

        onsets = np.vstack(onsets)
        cond_id = np.concatenate(cond_id)

        events.append((onsets, cond_id))

    return events


def make_design_matrices(events, n_scans, tr, hrf_model='canonical',
                         drift_model='cosine', motion=None):

    design_matrices = []
    n_sessions = len(n_scans)

    for i in range(n_sessions):

        onsets = events[i][0][:, 0]
        duration = events[i][0][:, 1]
        amplitude = events[i][0][:, 2]
        cond_id = events[i][1]
        order = np.argsort(onsets)

        # make a block or event paradigm depending on stimulus duration
        if duration.sum() == 0:
            paradigm = EventRelatedParadigm(cond_id[order],
                                            onsets[order],
                                            amplitude[order])
        else:
            paradigm = BlockParadigm(cond_id[order], onsets[order],
                                     duration[order], amplitude[order])

        frametimes = np.linspace(0, (n_scans[i] - 1) * tr, n_scans[i])

        if motion is not None:
            add_regs = np.array(motion[i]).astype('float')
            add_reg_names = ['motion_%i' % r
                             for r in range(add_regs.shape[1])]

            design_matrix = make_dmtx(
                frametimes, paradigm, hrf_model=hrf_model,
                drift_model=drift_model,
                add_regs=add_regs, add_reg_names=add_reg_names)
        else:
            design_matrix = make_dmtx(
                frametimes, paradigm, hrf_model=hrf_model,
                drift_model=drift_model)

        design_matrices.append(design_matrix.matrix)

    return design_matrices


def load_openfmri(study_dir, model_id,
                  hrf_model='canonical with derivative',
                  drift_model='cosine', glm_model='ar1',
                  n_jobs=-1, verbose=1):

    if n_jobs == 1:
        docs = []
        for subject_id in get_subjects_id(study_dir):
            docs.append(_load_openfmri(study_dir, subject_id, model_id,
                                       hrf_model, drift_model, glm_model,
                                       verbose - 1))
    else:
        docs = Parallel(n_jobs=n_jobs)(delayed(
            _load_openfmri)(study_dir, subject_id, model_id,
                            hrf_model, drift_model, glm_model, verbose - 1)
            for subject_id in get_subjects_id(study_dir))

    return docs


def _load_openfmri(study_dir, subject_id, model_id,
                   hrf_model, drift_model, glm_model, verbose):
    doc = {}
    subject_dir = os.path.join(study_dir, subject_id)

    doc['tr'] = get_scan_key(study_dir)['TR']
    doc['data'] = {}
    doc['data']['swabold'] = get_scans(subject_dir)
    doc['n_scans'] = get_scans_count(subject_dir)
    doc['motion'] = get_motion(subject_dir)
    doc['task_contrasts'] = get_task_contrasts(
        study_dir, subject_dir, model_id, hrf_model)

    events = get_events(subject_dir)

    doc['design_matrices'] = make_design_matrices(
        events, doc['n_scans'], doc['tr'],
        hrf_model, drift_model, doc['motion'])

    return doc
