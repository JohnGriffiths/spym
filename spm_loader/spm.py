import os
import glob
import hashlib
import multiprocessing

import numpy as np

from utils import fix_experiment
from utils import load_mat
from utils import find_data_dir
from utils import prefix_filename, strip_prefix_filename


def _check_kwargs(work_dir, **kwargs):
    doc = {}
    for k in kwargs:
        if hasattr(kwargs[k], '__call__'):
            doc[k] = kwargs[k](work_dir)
        elif isinstance(kwargs[k], int):
            doc[k] = work_dir.split(os.path.sep)[kwargs[k]]
        else:
            doc[k] = kwargs[k]
    return doc


def load_spm_design_matrix(design_matrix, n_scans, conditions):
    design_matrices = []
    n_sessions = len(n_scans)
    design_matrix = np.array(design_matrix)[:, :-n_sessions]
    conditions = np.array(conditions)[:-n_sessions]

    for i, dm in enumerate(np.vsplit(design_matrix, np.cumsum(n_scans[:-1]))):
        conditions_mask = np.array([
            'Sn(%s)' % (i + 1) in c for c in conditions])
        design_matrices.append(dm[:, conditions_mask])

    return design_matrices


def load_preproc(location, **kwargs):
    """ Loads the input parameters of the different processing steps of
        the SPM preprocessing.

        Parameters
        ----------
        location: str
            jobs_preproc.mat file path
        kwargs: dict
            extra args

        Note
        ----
        The returned parameters and paths are inputs of the steps, therefore
        the bold series in slice_timing will be the original time series,
        and the series in realign will be the resulting images from the
        slice_timing step.

        Returns
        -------
        A dict with one key per step.
    """

    doc = {}

    work_dir = os.path.split(os.path.realpath(location))[0]

    doc.update(_check_kwargs(work_dir, **kwargs))

    mat = load_mat(location)['jobs']

    doc['order'] = ['slice_timing', 'realign', 'preproc', 'normalize', 'final']

    def get_images(images, return_motion=False):
        if len(images[0].shape) == 0:
            data_dir = find_data_dir(work_dir, str(images[0]))
            paths = [[os.path.join(
                data_dir, os.path.split(str(x))[1]) for x in images]]
            fname = glob.glob(os.path.join(data_dir, 'rp_*.txt'))[0]
            motion = [fname]
        else:
            paths = []
            motion = []
            for session_scans in images:
                scans = []
                data_dir = find_data_dir(work_dir, str(session_scans[0]))
                for x in session_scans:
                    scans.append(
                        os.path.join(data_dir, os.path.split(str(x))[1]))
                paths.append(scans)
                fname = glob.glob(os.path.join(data_dir, 'rp_*.txt'))[0]
                motion.append(fname)

        if return_motion:
            return paths, motion
        return paths

    def get_path(path):
        data_dir = find_data_dir(work_dir, str(path))
        return os.path.join(data_dir, os.path.split(str(path))[1])

    # doc['mat'] = mat

    def save_slice_timing(m):
        # slice timing
        slice_timing = m
        doc['slice_timing'] = {}
        doc['slice_timing']['bold'] = []
        doc['slice_timing']['n_slices'] = int(slice_timing.nslices)
        doc['slice_timing']['ref_slice'] = int(slice_timing.refslice)
        doc['slice_timing']['slices_order'] = slice_timing.so.tolist()
        doc['slice_timing']['ta'] = float(slice_timing.ta)
        doc['slice_timing']['tr'] = float(slice_timing.tr)

        doc['slice_timing']['bold'] = get_images(slice_timing.scans)

        n_scans = [len(s) for s in doc['slice_timing']['bold']]
        doc['n_scans'] = n_scans

    def save_realign(m0):
        # realign
        realign = m0

        doc['realign'] = {}
        doc['realign']['bold'] = []
        doc['realign']['motion'] = []

        doc['realign']['bold'], doc['realign']['motion'] = \
            get_images(realign.data, return_motion=True)

    def save_preproc_normalize(m0, m1):
        # preproc / normalize
        preproc = m0
        normalize = m1

        doc['preproc'] = {}
        doc['normalize'] = {}
        doc['preproc']['anatomy'] = get_path(preproc.data)
        doc['normalize']['anatomy'] = get_path(normalize.write.subj.resample)
        doc['normalize']['mat_file'] = get_path(normalize.write.subj.matname)

    def save_coregistration(m0):
        # coregistration
        coreg = m0

        doc['coregistration'] = {}
        doc['coregistration']['realigned'] = [get_path(p)
                                              for p in coreg.estimate.other]
        doc['coregistration']['anatomy'] = get_path(coreg.estimate.ref)
        doc['coregistration']['realigned_ref'] = get_path(
            coreg.estimate.source)

    def save_normalize(m0):
        # normalize bold
        normalize = m0

        doc['normalize']['bold'] = []

        for session_scans in np.split(
                normalize.resample, np.cumsum(doc['n_scans']))[:-1]:
            scans = []
            data_dir = find_data_dir(work_dir, str(session_scans[0]))

            for x in session_scans:
                scans.append(os.path.join(data_dir, os.path.split(str(x))[1]))
            doc['normalize']['bold'].append(scans)

    def save_smooth(m0):
        # smooth
        smooth = m0

        doc['smooth'] = {}
        doc['smooth']['bold'] = []
        doc['smooth']['fwhm'] = float(smooth.fwhm)

        for session_scans in np.split(
                smooth.data, np.cumsum(doc['n_scans']))[:-1]:
            scans = []
            data_dir = find_data_dir(work_dir, str(session_scans[0]))

            for x in session_scans:
                scans.append(os.path.join(data_dir, os.path.split(str(x))[1]))
            doc['smooth']['bold'].append(scans)

    for m in mat:
        try:
            m0 = m.temporal.st
            save_slice_timing(m0)
        except:
            pass

        try:
            m0 = m.spatial.realign.estwrite
            save_realign(m0)
        except:
            pass

        try:
            m0 = m.spatial[0].preproc
            m1 = m.spatial[1].normalise
            save_preproc_normalize(m0, m1)
        except:
            pass

        try:
            m0 = m.spatial.coreg
            save_coregistration(m0)
        except:
            pass

        try:
            m0 = m.spatial.normalise.write.subj
            save_normalize(m0)
        except:
            pass

        try:
            m0 = m.spatial.smooth
            save_smooth(m0)
        except:
            pass

    doc['final'] = {}
    doc['final']['anatomy'] = prefix_filename(
        doc['normalize']['anatomy'], 'w')

    doc['final']['bold'] = []

    for session_scans in doc['smooth']['bold']:
        scans = []
        for x in session_scans:
            scans.append(prefix_filename(x, 's'))
        doc['final']['bold'].append(scans)

    if 'subject_id' not in doc:
        doc['subject_id'] = hashlib.md5(work_dir).hexdigest()

    return doc


def load_intra(location, fix=None, **kwargs):
    doc = {}

    mat = load_mat(location)['SPM']

    work_dir = os.path.split(os.path.realpath(location))[0]

    doc.update(_check_kwargs(work_dir, **kwargs))

    # doc['mat'] = mat

    doc['design_matrix'] = mat.xX.X.tolist()           # xX: model
    doc['design_matrix_conditions'] = [str(i) for i in mat.xX.name]
    doc['design_matrix_contrasts'] = {}

    doc['n_scans'] = mat.nscan.tolist() \
        if isinstance(mat.nscan.tolist(), list) else [mat.nscan.tolist()]
    doc['n_sessions'] = mat.nscan.size
    doc['tr'] = float(mat.xY.RT)    # xY: data
    doc['mask'] = os.path.join(work_dir, str(mat.VM.fname))  # VM: mask

    doc['beta_maps'] = []
    doc['c_maps'] = {}
    doc['t_maps'] = {}

    doc['condition_key'] = []
    doc['task_contrasts'] = {}
    doc['onsets'] = []

    swabold = np.split(mat.xY.P.tolist(), np.cumsum(doc['n_scans'])[:-1])

    doc['data'] = {}

    for session in swabold:
        session_dir = find_data_dir(work_dir, session[0])
        scans = [os.path.join(session_dir, os.path.split(s)[1].strip())
                 for s in session]
        doc['data'].setdefault('swabold', []).append(scans)

    for s in doc['data']['swabold']:
        scans = []
        for i in s:
            scans.append(strip_prefix_filename(i, 1))
        doc['data'].setdefault('wabold', []).append(scans)

    for s in doc['data']['swabold']:
        scans = []
        for i in s:
            scans.append(strip_prefix_filename(i, 2))
        doc['data'].setdefault('abold', []).append(scans)

    for s in doc['data']['swabold']:
        scans = []
        for i in s:
            scans.append(strip_prefix_filename(i, 3))
        doc['data'].setdefault('bold', []).append(scans)

    doc['motion'] = []
    if doc['n_sessions'] > 1:
        for session in mat.Sess:
            doc['motion'].append(session.C.C.tolist())
    else:
        doc['motion'].append(mat.Sess.C.C.tolist())

    def get_condition_onsets(condition):
        onset_time = condition.ons.tolist()
        onset_duration = condition.dur.tolist()
        if not isinstance(onset_time, list):
            onset_time = [onset_time]
            onset_duration = [onset_duration]
        onset_weight = [1] * len(onset_time)

        return zip(onset_time, onset_duration, onset_weight)

    if hasattr(mat.Sess, '__iter__'):
        for session in mat.Sess:
            onsets = {}
            condition_key = []
            for condition_id, condition in enumerate(session.U):
                k = 'cond%03i' % (condition_id + 1)
                onsets[k] = get_condition_onsets(condition)
                condition_key.append(str(condition.name))
            doc['condition_key'].append(condition_key)
            doc['onsets'].append(onsets)
    else:
        onsets = {}
        condition_key = []
        for condition_id, condition in enumerate(mat.Sess.U):
            k = 'cond%03i' % (condition_id + 1)
            onsets[k] = get_condition_onsets(condition)
            condition_key.append(str(condition.name))
        doc['condition_key'].append(condition_key)
        doc['onsets'].append(onsets)

    for c in mat.xCon:
        name = str(c.name)

        doc['c_maps'][name] = os.path.join(work_dir, str(c.Vcon.fname))
        doc['t_maps'][name] = os.path.join(work_dir, str(c.Vspm.fname))
        doc['design_matrix_contrasts'][name] = c.c.tolist()

    for i, b in enumerate(mat.Vbeta):
        doc['beta_maps'].append(os.path.join(work_dir, str(b.fname)))

    if 'subject_id' not in doc:
        doc['subject_id'] = hashlib.md5(work_dir).hexdigest()

    def get_condition_index(name):
        for i, full_name in enumerate(doc['design_matrix_conditions']):
            if name in full_name:
                return i

    # find the indices of the actual experimental conditions in the
    # design matrix, not the additional regressors...
    ii = []
    for session in doc['condition_key']:
        ii.append([get_condition_index(name) for name in session])
    # redefine the contrasts with the experimental conditions
    for k, contrast in doc['design_matrix_contrasts'].iteritems():
        doc['task_contrasts'][k] = []

        for per_session in ii:
            doc['task_contrasts'][k].append(
                np.array(contrast)[per_session].tolist())

    # attempt to guess condition names with the contrast names & values
    condition_key = [np.array(ck, dtype='|S32') for ck in doc['condition_key']]
    for contrast_name, session_contrasts in doc['task_contrasts'].items():
        for ck, contrast in zip(condition_key, session_contrasts):
            contrast = np.array(contrast)
            if ((contrast < 0).sum() == 0 and len(contrast.shape) == 1
                and (contrast == np.abs(contrast).max()).sum() == 1):
                ck[np.array(contrast) > 0] = contrast_name
    doc['condition_key'] = condition_key
    if fix is not None:
        doc = fix_experiment(doc, fix)[0]
    return doc


def load_dotmat_files(data_dir, study_id, subjects_id,
                      dotmat_relpath, load_dotmat,
                      get_subject=None, n_jobs=-1, **kwargs):
    docs = []
    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    if n_jobs == 1:
        for subject_id in subjects_id:
            subject_dir = os.path.join(data_dir, subject_id)
            kwds = dict(subject_id=get_subject, study_id=study_id)
            kwds.update(kwargs)
            mat_file = os.path.join(subject_dir, dotmat_relpath)
            docs.append(load_dotmat(mat_file, **kwds))
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        for subject_id in subjects_id:
            subject_dir = os.path.join(data_dir, subject_id)
            kwds = dict(subject_id=get_subject, study_id=study_id)
            kwds.update(kwargs)
            mat_file = os.path.join(subject_dir, dotmat_relpath)
            ar = pool.apply_async(load_dotmat,
                                  args=(mat_file, ),
                                  kwds=kwds)
            docs.append(ar)

        pool.close()
        pool.join()

        docs = [doc.get() for doc in docs]

    return docs
