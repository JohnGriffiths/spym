import os
import hashlib
import gzip
import os.path as pt

import scipy.io as sio
import numpy as np


def load_mat(location):
    if location.endswith('.gz'):
        return sio.loadmat(
            gzip.open(location, 'rb'),
            squeeze_me=True,
            struct_as_record=False
            )['SPM']

    return sio.loadmat(
        location, squeeze_me=True, struct_as_record=False)


def _wdir(wd):
    def func(path):
        return pt.join(str(wd), pt.split(str(path))[1])
    return func


def _find_data_dir(wd, fpath):

    def right_splits(p):
        while p not in ['', None]:
            p = p.rsplit(pt.sep, 1)[0]
            yield p

    def left_splits(p):
        while len(p.split(pt.sep, 1)) > 1:
            p = p.split(pt.sep, 1)[1]
            yield p

    if not pt.isfile(fpath):
        for rs in right_splits(wd):
            for ls in left_splits(fpath):
                p = pt.join(rs, *ls.split(pt.sep))
                if pt.isfile(p):
                    return pt.dirname(p)
    else:
        return pt.dirname(fpath)


def _prefix_filename(path, prefix):
    path, filename = pt.split(str(path))
    return pt.join(path, '%s%s' % (prefix, filename))


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

    doc['order'] = ['slice_timing', 'realign', 'preproc', 'normalise', 'final']

    # slice timing
    slice_timing = mat[0].temporal.st

    doc['slice_timing'] = {}
    doc['slice_timing']['bold'] = []
    doc['slice_timing']['n_slices'] = int(slice_timing.nslices)
    doc['slice_timing']['ref_slice'] = int(slice_timing.refslice)
    doc['slice_timing']['slices_order'] = slice_timing.so.tolist()
    doc['slice_timing']['ta'] = float(slice_timing.ta)
    doc['slice_timing']['tr'] = float(slice_timing.tr)

    for session_scans in slice_timing.scans:
        scans = []
        data_dir = _find_data_dir(work_dir, str(session_scans[0]))

        for x in session_scans:
            scans.append(os.path.join(data_dir, os.path.split(str(x))[1]))
        doc['slice_timing']['bold'].append(scans)

    n_sessions = [len(s) for s in slice_timing.scans]
    doc['n_sessions'] = n_sessions

    # realign
    realign = mat[1].spatial.realign.estwrite

    doc['realign'] = {}
    doc['realign']['bold'] = []

    for session_scans in realign.data:
        scans = []
        data_dir = _find_data_dir(work_dir, str(session_scans[0]))

        for x in session_scans:
            scans.append(os.path.join(data_dir, os.path.split(str(x))[1]))
        doc['realign']['bold'].append(scans)

    # preproc / normalise
    preproc = mat[2].spatial[0].preproc
    normalise = mat[2].spatial[1].normalise

    doc['preproc'] = {}
    doc['normalise'] = {}
    doc['preproc']['anatomy'] = str(preproc.data)
    doc['normalise']['anatomy'] = str(normalise.write.subj.resample)
    doc['normalise']['mat_file'] = str(normalise.write.subj.matname)

    # coregistration
    # mat[3].spatial.coreg.estimate.other > realigned bold
    # mat[3].spatial.coreg.estimate.ref > first realigned bold volume
    # mat[3].spatial.coreg.estimate.source > normalised anatomy

    # normalise bold

    normalise = mat[4].spatial.normalise.write.subj

    doc['normalise']['bold'] = []

    for session_scans in np.split(
            normalise.resample, np.cumsum(n_sessions))[:-1]:
        scans = []
        data_dir = _find_data_dir(work_dir, str(session_scans[0]))

        for x in session_scans:
            scans.append(os.path.join(data_dir, os.path.split(str(x))[1]))
        doc['normalise']['bold'].append(scans)

    # smooth
    smooth = mat[5].spatial.smooth

    doc['smooth'] = {}
    doc['smooth']['bold'] = []
    doc['smooth']['fwhm'] = float(smooth.fwhm)

    for session_scans in np.split(smooth.data, np.cumsum(n_sessions))[:-1]:
        scans = []
        data_dir = _find_data_dir(work_dir, str(session_scans[0]))

        for x in session_scans:
            scans.append(os.path.join(data_dir, os.path.split(str(x))[1]))
        doc['smooth']['bold'].append(scans)

    doc['final'] = {}
    doc['final']['anatomy'] = _prefix_filename(
        doc['normalise']['anatomy'], 'w')

    doc['final']['bold'] = []

    for session_scans in doc['smooth']['bold']:
        scans = []
        for x in session_scans:
            scans.append(_prefix_filename(x, 's'))
        doc['final']['bold'].append(scans)

    if 'subject_id' not in doc:
        doc['subject_id'] = hashlib.md5(data_dir).hexdigest()

    return doc


def load_intra(location, inputs=True, outputs=True, **kwargs):
    spmmat = load_mat(location)['SPM']

    wd, _ = pt.split(pt.realpath(location))  # work dir
    bd = _wdir(wd)                           # beta directory

    analysis = {}

    for k in kwargs:
        if hasattr(kwargs[k], '__call__'):
            analysis[k] = kwargs[k](wd)
        elif isinstance(kwargs[k], int):
            analysis[k] = wd.split(pt.sep)[kwargs[k]]
        else:
            analysis[k] = kwargs[k]

    analysis['design_matrix'] = spmmat.xX.X.tolist()           # xX: design
    analysis['conditions'] = [str(i) for i in spmmat.xX.name]  # xX: design
    analysis['n_scans'] = spmmat.nscan.tolist() \
        if isinstance(spmmat.nscan.tolist(), list) else [spmmat.nscan.tolist()]
    analysis['n_sessions'] = spmmat.nscan.size
    analysis['TR'] = float(spmmat.xY.RT)    # xY: data
    analysis['mask'] = bd(spmmat.VM.fname)  # VM: mask

    if outputs:
        analysis['beta_maps'] = []
        analysis['c_maps'] = {}
        analysis['c_maps_smoothed'] = {}
        analysis['t_maps'] = {}
        analysis['contrast_definitions'] = {}

        for c in spmmat.xCon:
            name = str(c.name)
            scon = _prefix_filename(c.Vcon.fname, 's')

            analysis['c_maps'][name] = bd(c.Vcon.fname)
            analysis['c_maps_smoothed'][name] = bd(scon)
            analysis['t_maps'][name] = bd(c.Vspm.fname)
            analysis['contrast_definitions'][name] = c.c.tolist()

        for i, b in enumerate(spmmat.Vbeta):
            analysis['beta_maps'].append(bd(b.fname))

    if inputs:
        analysis['anatomy'] = []
        analysis['swa_anatomy'] = []
        analysis['bold'] = []
        analysis['swa_bold'] = []
        analysis['wa_bold'] = []
        analysis['a_bold'] = []
        for Y in spmmat.xY.P:
            Y = str(Y).strip()
            data_dir = _find_data_dir(wd, Y)
            if data_dir is not None:
                analysis['swa_anatomy'].append(
                    pt.join(data_dir, pt.split(Y)[1]))
                analysis['anatomy'].append(
                    pt.join(data_dir, pt.split(Y)[1].strip('swa')))
                analysis['bold'].append(
                    pt.join(data_dir, pt.split(Y)[1].strip('swa')))
                analysis['swa_bold'].append(
                    pt.join(data_dir, 'swa%s' % pt.split(Y)[1].strip('swa')))
                analysis['wa_bold'].append(
                    pt.join(data_dir, 'wa%s' % pt.split(Y)[1].strip('swa')))
                analysis['a_bold'].append(
                    pt.join(data_dir, 'a%s' % pt.split(Y)[1].strip('swa')))
            else:
                analysis['anatomy'].append(pt.split(Y)[1].strip('swa'))
                analysis['swa_anatomy'].append(pt.split(Y)[1])
                analysis['bold'].append(pt.split(Y)[1].strip('swa'))
                analysis['swa_bold'].append(
                    'swa%s' % pt.split(Y)[1].strip('swa'))
                analysis['wa_bold'].append(
                    'wa%s' % pt.split(Y)[1].strip('swa'))
                analysis['a_bold'].append(
                    'a%s' % pt.split(Y)[1].strip('swa'))

    return analysis
