import os
import glob
import json
import shutil
import fnmatch
import multiprocessing

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.modalities.fmri.experimental_paradigm import EventRelatedParadigm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm
from joblib import Parallel, delayed

from utils import remove_special
from glm import _first_level_glm


# ----------------------------------------------------------------------------
# parse openfmri layout
# ----------------------------------------------------------------------------

def get_study_tr(study_dir):
    return float(
        open(os.path.join(study_dir, 'scan_key.txt')).read().split()[1])


def get_task_sessions(subject_dir):
    sessions = os.path.join(subject_dir, 'BOLD', '*')
    return [os.path.split(session)[1].split('_')[0]
            for session in sorted(glob.glob(sessions))]


def get_motion(subject_dir):
    sessions = os.path.join(subject_dir, 'BOLD', '*')

    motion = []
    for session_dir in sorted(glob.glob(sessions)):
        motion_s = open(os.path.join(session_dir, 'motion.txt')).read()
        motion_s = np.array([l.split() for l in motion_s.split('\n')][:-1])
        motion.append(np.array(motion_s).astype('float'))
    return motion


def get_bold_images(subject_dir):
    sessions = os.path.join(subject_dir, 'BOLD', '*')

    images = []
    for session_dir in sorted(glob.glob(sessions)):
        img = nb.load(os.path.join(session_dir, 'normalized_bold.nii.gz'))
        images.append(img)

    n_scans = [img.shape[-1] for img in images]

    return images, n_scans


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


# ----------------------------------------------------------------------------
# dump openfmri layout from spm
# ----------------------------------------------------------------------------

def write_new_model(study_dir, model_id, contrasts):
    models_dir = os.path.join(study_dir, 'models')

    if not os.path.exists(os.path.join(models_dir, model_id)):
        os.makedirs(os.path.join(models_dir, model_id))

    cond_model001 = os.path.join(models_dir, 'model001', 'condition_key.txt')
    cond_model002 = os.path.join(models_dir, model_id, 'condition_key.txt')

    shutil.copyfile(cond_model001, cond_model002)

    contrasts_path = os.path.join(models_dir, model_id, 'task_contrasts.txt')

    with open(contrasts_path, 'wb') as f:
        for contrast in contrasts:
            task_id, contrast_id = contrast.split('__')
            con_val = contrasts[contrast]
            con_val = ' '.join(np.array(con_val).astype('|S32'))
            f.write('%s %s %s\n' % (task_id, contrast_id, con_val))


def spm_to_openfmri(out_dir, preproc_docs, intra_docs, metadata=None,
                    n_jobs=-1, verbose=1):
    metadata = _check_metadata(metadata, preproc_docs[0], intra_docs[0])

    _openfmri_metadata(os.path.join(out_dir, metadata['study_id']), metadata)

    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    docs = zip(preproc_docs, intra_docs)

    if n_jobs == 1:
        for i, (preproc_doc, intra_doc) in enumerate(docs):
            _openfmri_preproc(out_dir, preproc_doc, metadata, verbose)
            _openfmri_intra(out_dir, intra_doc, metadata, verbose)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        for i, (preproc_doc, intra_doc) in enumerate(docs):
            pool.apply_async(_openfmri_preproc,
                             args=(out_dir, preproc_doc, metadata, verbose))
            pool.apply_async(_openfmri_intra,
                             args=(out_dir, intra_doc, metadata, verbose))
        pool.close()
        pool.join()


def _check_metadata(metadata, preproc_doc, intra_doc):
    if metadata is None:
        metadata = {}

    if not 'run_key' in metadata:
        raise Exception('Need a run_key in metadata')

    if not 'condition_key' in metadata:
        metadata['condition_key'] = {}
        for run_key, conditions in zip(metadata['run_key'],
                                       intra_doc['condition_key']):
            metadata['condition_key'][run_key] = conditions

    if not 'scan_key' in metadata:
        metadata['scan_key'] = {}
        metadata['scan_key']['TR'] = intra_doc['tr']

    if 'study_id' in intra_doc:
        metadata['study_id'] = intra_doc['study_id']
    else:
        metadata['study_id'] = ''

    return metadata


def _openfmri_preproc(out_dir, doc, metadata=None, verbose=1):
    """
        Parameters
        ----------
        metadata: dict
            - run_key: naming the sessions

        Examples
        --------
        {'run_key': ['task001 run001', 'task001 run002',
                     'task002 run001', 'task002 run002']}

    """
    if 'study_id' in doc:
        study_dir = os.path.join(out_dir, doc['study_id'])
    else:
        study_dir = out_dir

    if verbose > 0:
        print '%s@%s: dumping preproc' % (doc['subject_id'], doc['study_id'])

    subject_dir = os.path.join(study_dir, doc['subject_id'])
    anatomy_dir = os.path.join(subject_dir, 'anatomy')

    if not os.path.exists(anatomy_dir):
        os.makedirs(anatomy_dir)

    anatomy = doc['preproc']['anatomy']
    wm_anatomy = doc['final']['anatomy']

    anatomy = nb.load(anatomy)
    wm_anatomy = nb.load(wm_anatomy)

    nb.save(anatomy, os.path.join(anatomy_dir, 'highres001.nii.gz'))
    nb.save(wm_anatomy, os.path.join(anatomy_dir,
                                     'normalized_highres001.nii.gz'))

    bold_dir = os.path.join(subject_dir, 'BOLD')

    for session, run_key in zip(doc['slice_timing']['bold'],
                                metadata['run_key']):

        bold = nb.concat_images(session)
        session_dir = os.path.join(bold_dir, run_key.replace(' ', '_'))
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        nb.save(bold, os.path.join(session_dir, 'bold.nii.gz'))

    for session, motion, run_key in zip(doc['final']['bold'],
                                        doc['realign']['motion'],
                                        metadata['run_key']):

        bold = nb.concat_images(session)
        session_dir = os.path.join(bold_dir, run_key.replace(' ', '_'))
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        nb.save(bold, os.path.join(session_dir, 'normalized_bold.nii.gz'))
        shutil.copyfile(motion, os.path.join(session_dir, 'motion.txt'))


def _openfmri_intra(out_dir, doc, metadata=None, verbose=1):
    """
        Parameters
        ----------
        metadata: dict
            - condition_key
              https://openfmri.org/content/metadata-condition-key

        Examples
        --------
        {'condition_key': {'task001 cond001': 'task',
                           'task001 cond002': 'parametric gain'}}
    """
    if 'study_id' in doc:
        study_dir = os.path.join(out_dir, doc['study_id'])
    else:
        study_dir = out_dir

    if verbose > 0:
        print '%s@%s: dumping stats intra' % (doc['subject_id'],
                                              doc['study_id'])

    subject_dir = os.path.join(study_dir, doc['subject_id'])

    model_dir = os.path.join(study_dir, 'models', 'model001')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # conditions specification
    conditions_spec = []
    for key, val in sorted(metadata['condition_key'].iteritems()):
        for i, name in enumerate(val):
            conditions_spec.append(
                '%s cond%03i %s\n' % (key.split(' ')[0], i + 1, name))

    with open(os.path.join(model_dir, 'condition_key.txt'), 'wb') as f:
                f.write(''.join(sorted(set(conditions_spec))))

    # contrasts specification
    contrasts_spec = []
    for key, val in doc['task_contrasts'].iteritems():
        if 'task_contrasts' in metadata:
            key = doc['task_contrasts'][key]

        for i, session_contrast in enumerate(val):
            task_id = metadata['run_key'][i].split(' ')[0]
            # check not null and 1d
            if (np.abs(session_contrast).sum() > 0
                and len(np.array(session_contrast).shape) == 1):
                con = ' '.join(np.array(session_contrast).astype('|S32'))
                contrasts_spec.append('%s %s %s\n' % (task_id, key, con))

    with open(os.path.join(model_dir, 'task_contrasts.txt'), 'wb') as f:
        f.write(''.join(sorted(set(contrasts_spec))))

    # dump onsets
    model_dir = os.path.join(subject_dir, 'model', 'model001')
    onsets_dir = os.path.join(model_dir, 'onsets')

    for onsets, run_key in zip(doc['onsets'], metadata['run_key']):
        run_dir = os.path.join(onsets_dir, run_key.replace(' ', '_'))
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        for condition_id, values in onsets.iteritems():
            cond = os.path.join(run_dir, '%s.txt' % condition_id)
            with open(cond, 'wb') as f:
                for timepoint in values:
                    f.write('%s %s %s\n' % timepoint)

    # analyses
    for dtype in ['c_maps', 't_maps']:
        data_dir = os.path.join(model_dir, dtype)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if isinstance(doc[dtype], dict):
            for contrast_id in doc[dtype].keys():
                fname = remove_special(contrast_id)
                img = nb.load(doc[dtype][contrast_id])
                nb.save(img, os.path.join(data_dir, '%s.nii.gz' % fname))

    # general data for analysis
    img = nb.load(doc['mask'])
    nb.save(img, os.path.join(model_dir, 'mask.nii.gz'))
    json.dump(doc, open(os.path.join(model_dir, 'SPM.json'), 'wb'))


def _openfmri_metadata(out_dir, metadata):
    """ General dataset information

        Parameters
        ----------
        metadata: dict
            - task_key -- https://openfmri.org/content/metadata-task-key
            - scan_key -- https://openfmri.org/content/metadata-scan-key

        Examples
        --------
        {'task_key': {'task001': 'stop signal with manual response',
                      'task002': 'stop signal with letter naming'}}
        {'scan_key': {'TR': 2.0}
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # naming the tasks
    if 'task_key' in metadata:
        with open(os.path.join(out_dir, 'task_key.txt'), 'wb') as f:
            for key, val in sorted(metadata['task_key'].iteritems()):
                f.write('%s %s\n' % (key, val))

    # scanning info
    if 'scan_key' in metadata:
        with open(os.path.join(out_dir, 'scan_key.txt'), 'wb') as f:
            for key, val in sorted(metadata['scan_key'].iteritems()):
                f.write('%s %s\n' % (key, val))

    # extra info, for example subject_id mapping etc...
    if 'extras' in metadata:
        meta_dir = os.path.join(out_dir, 'metadata')
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir)
        for key, val in metadata['extras'].iteritems():
            with open(os.path.join(meta_dir, '%s.txt' % key), 'wb') as f:
                for k, v in sorted(val.iteritems()):
                    f.write('%s %s\n' % (k, v))


# ----------------------------------------------------------------------------
# GLM on openfmri layout
# ----------------------------------------------------------------------------


def first_level_glm(study_dir, subjects_id, model_id,
                     hrf_model='canonical with derivative',
                     drift_model='cosine',
                     glm_model='ar1', mask='compute', n_jobs=-1, verbose=1):
    """ Utility function to compute first level GLMs in parallel
    """

    if n_jobs == 1:
        for subject_id in subjects_id:
            _openfmri_first_level_glm(study_dir, subject_id, model_id,
                                      hrf_model, drift_model, glm_model,
                                      mask, verbose - 1)
    else:
        Parallel(n_jobs=n_jobs)(delayed(
            _openfmri_first_level_glm)(
                study_dir, subject_id, model_id,
                hrf_model, drift_model, glm_model, mask, verbose - 1)
                for subject_id in subjects_id
            )


def _openfmri_first_level_glm(study_dir, subject_id, model_id,
                              hrf_model='canonical with derivative',
                              drift_model='cosine',
                              glm_model='ar1', mask='compute', verbose=1):
    study_id = os.path.split(study_dir)[1]
    subject_dir = os.path.join(study_dir, subject_id)

    if verbose > 0:
        print '%s@%s: first level glm' % (subject_id, study_id)

    tr = get_study_tr(study_dir)
    images, n_scans = get_bold_images(subject_dir)
    motion = get_motion(subject_dir)
    contrasts = get_task_contrasts(study_dir, subject_dir, model_id, hrf_model)
    events = get_events(subject_dir)

    design_matrices = make_design_matrices(events, n_scans, tr,
                                           hrf_model, drift_model, motion)

    import pylab as pl

    for session_id, dm in enumerate(design_matrices):
        pl.matshow(dm)
        pl.savefig('/tmp/%s_%s.png' % (subject_id, session_id + 1), dpi=300)

    model_dir = os.path.join(study_dir, subject_id, 'model', model_id)

    _first_level_glm(model_dir, images, design_matrices,
                     contrasts, glm_model, mask, verbose - 1)
