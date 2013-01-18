import os
import json
import multiprocessing

import nibabel as nb
import numpy as np

from nipy.modalities.fmri.glm import FMRILinearModel


def fix_docs(docs, fix=None, fields=None):
    if fields is None:
        fields = ['t_maps', 'c_maps', 'c_maps_smoothed', 'contrasts']
    if fix is None or fix == {}:
        return docs
    fixed_docs = []

    for doc in docs:
        fixed_doc = {}

        for key in doc.keys():
            if key not in fields:
                fixed_doc[key] = doc[key]

        for field in fields:

            for name in doc[field].keys():
                if name in fix.keys():
                    fixed_doc.setdefault(
                        field,
                        {}).setdefault(fix[name], doc[field][name])

        fixed_docs.append(fixed_doc)

    return fixed_docs


def export(docs, out_dir, fix=None, outputs=None, n_jobs=1):
    """ Export data described in documents to fixed folder structure.

        e.g. {out_dir}/{study_name}/subjects/{subject_id}/c_maps/...

        Parameters
        ----------
        out_dir: string
            Destination directory.
        docs: list
            List of documents
        fix: dict
            Map names for c_maps and t_maps
        outputs: dict
            Data to export, default is True for all.
            Possible keys are 'maps', 'data', 'mask', 'model'
            e.g. outputs = {'maps': False} <=> export all but maps
    """
    docs = fix_docs(docs, fix)

    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    if n_jobs == 1:
        for doc in docs:
            _export(doc, out_dir, outputs)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        for doc in docs:
            pool.apply_async(_export, args=(doc, out_dir, outputs))
        pool.close()
        pool.join()


def _export(doc, out_dir, outputs):
    study_dir = os.path.join(out_dir, doc['study'])
    subject_dir = os.path.join(study_dir, 'subjects', doc['subject'])

    if not os.path.exists(study_dir):
        os.makedirs(study_dir)
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)

    if outputs is None:
        outputs = {}
    # maps
    if outputs.get('maps', True):
        for dtype in ['t_maps', 'c_maps']:
            map_dir = os.path.join(subject_dir, dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            for label, fpath in doc[dtype].iteritems():
                img = nb.load(fpath)
                fname = '%s.nii.gz' % label.replace(' ', '_')
                nb.save(img, os.path.join(map_dir, fname))

    # fMRI
    if 'data' in doc and outputs.get('data', True):
        for dtype in ['raw_data', 'data']:
            img = nb.concat_images(doc[dtype])
            fname = 'bold' if dtype == 'data' else 'raw_bold'
            nb.save(img, os.path.join(subject_dir, '%s.nii.gz' % fname))

    # mask
    if outputs.get('mask', True):
        img = nb.load(doc['mask'])
        nb.save(img, os.path.join(subject_dir, 'mask.nii.gz'))

    # model
    if outputs.get('model', True):
        design_matrix = doc['design_matrix']
        path = os.path.join(subject_dir, 'design_matrix.json')
        json.dump(design_matrix, open(path, 'wb'))
        contrasts = doc['contrasts']
        path = os.path.join(subject_dir, 'contrasts.json')
        json.dump(contrasts, open(path, 'wb'))


def _get_timeseries(data, row_mask, affine=None):
    if isinstance(data, list):
        return nb.concat_images(np.array(data)[row_mask])
    elif isinstance(data, (str, unicode)):
        img = nb.load(data)
        return nb.Nifti1Image(img.get_data()[row_mask, :], img.get_affine())
    elif isinstance(data, (np.ndarray, np.memmap)):
        if affine is None:
            raise Exception('The affine is not optional '
                            'when data is an array')
        return nb.Nifti1Image(data[row_mask, :], affine)
    else:
        raise ValueError('Data type "%s" not supported' % type(data))


def load_glm_params(doc):
    params = {}

    n_scans = doc['n_scans']
    n_sessions = len(n_scans)

    design_matrix = np.array(doc['design_matrix'])[:, :-n_sessions]

    params['design_matrices'] = []
    params['contrasts'] = []
    params['data'] = []

    offset = 0
    for session_id, scans_count in enumerate(n_scans):
        session_dm = design_matrix[offset:offset + scans_count, :]
        column_mask = ~(np.sum(session_dm, axis=0) == 0)
        row_mask = np.zeros(np.sum(n_scans), dtype=np.bool)
        row_mask[offset:offset + scans_count] = True

        params['design_matrices'].append(session_dm[:, column_mask])

        session_contrasts = {}
        for contrast_id in doc['contrasts']:
            contrast = np.array(doc['contrasts'][contrast_id])
            session_contrast = contrast[column_mask]
            if not np.all(session_contrast == 0):
                session_contrast /= session_contrast.max()
            session_contrasts[contrast_id] = session_contrast
        params['contrasts'].append(session_contrasts)
        params['data'].append(_get_timeseries(doc['data'], row_mask))
        offset += scans_count

    return params


def make_contrasts(params, definitions):
    new_contrasts = []
    for old_session_contrasts in params['contrasts']:
        new_session_contrasts = {}
        for new_contrast_id in definitions:
            contrast = None
            for old_contrast_id in definitions[new_contrast_id]:
                scaler = definitions[new_contrast_id][old_contrast_id]
                con = np.array(old_session_contrasts[old_contrast_id]) * scaler
                if contrast is None:
                    contrast = con
                else:
                    contrast += con
            new_session_contrasts[new_contrast_id] = contrast
        new_contrasts.append(new_session_contrasts)
    return new_contrasts


def execute_glm(doc, out_dir, contrast_definitions=None,
                outputs=None, glm_model='ar1'):
    study_dir = os.path.join(out_dir, doc['study'])
    subject_dir = os.path.join(study_dir, 'subjects', doc['subject'])

    if outputs is None:
        outputs = {'maps': False, 'data': False}
    else:
        outputs['maps'] = False

    export([doc], out_dir, outputs=outputs)

    params = load_glm_params(doc)

    glm = FMRILinearModel(params['data'],
                          params['design_matrices'], doc['mask'])

    glm.fit(do_scaling=True, model=glm_model)

    if contrast_definitions is not None:
        params['contrasts'] = make_contrasts(params, contrast_definitions)
    contrasts = sorted(params['contrasts'][0].keys())

    for index, contrast_id in enumerate(contrasts):
        print ' study[%s] subject[%s] contrast [%s]: %i/%i' % (
            doc['study'], doc['subject'],
            contrast_id, index + 1, len(contrasts)
            )
        contrast = [c[contrast_id] for c in params['contrasts']]
        contrast_name = contrast_id.replace(' ', '_')
        z_map, t_map, c_map, var_map = glm.contrast(
            contrast,
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True,)

        for dtype, out_map in zip(['z', 't', 'c', 'variance'],
                                  [z_map, t_map, c_map, var_map]):
            map_dir = os.path.join(subject_dir, '%s_maps' % dtype)
            if not os.path.exists(map_dir):
                os.makedirs(map_dir)
            map_path = os.path.join(map_dir, '%s.nii.gz' % contrast_name)
            nb.save(out_map, map_path)


def execute_glms(docs, out_dir, contrast_definitions=None,
                 outputs=None, glm_model='ar1', n_jobs=1):

    n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    if n_jobs == 1:
        for doc in docs:
            execute_glm(doc, out_dir, contrast_definitions, outputs, glm_model)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
        for doc in docs:
            pool.apply_async(
                execute_glm,
                args=(doc, out_dir, contrast_definitions, outputs, glm_model))
        pool.close()
        pool.join()
