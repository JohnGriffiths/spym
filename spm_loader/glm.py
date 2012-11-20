import os
import json
import tempfile
import multiprocessing

import numpy as np
import nibabel as nb
import pylab as pl

from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.labs.viz import plot_map, cm

from .spm import load_intra


def get_fmri_data(data, indices):
    if isinstance(indices, tuple):
        indices = slice(*indices)
    if isinstance(data, list):
        return nb.concat_images(data[indices])
    elif isinstance(data, (str, unicode)):
        return nb.Nifti1Image(nb.load(data).get_data()[indices, :],
                              affine=nb.load(data).get_affine())


def get_frmi_runs(data, runs):
    runs = [slice(*run) if isinstance(run, tuple) else run for run in runs]
    if isinstance(data, list):
        return [nb.concat_images(data[run]) for run in runs]
    elif isinstance(data, (str, unicode)):
        return [nb.Nifti1Image(nb.load(data).get_data()[run, :],
                affine=nb.load(data).get_affine())
                for run in runs]


def fit_glm(data, design_matrices, mask, do_scaling, model):
    glm = FMRILinearModel(data, design_matrices, mask)
    glm.fit(do_scaling, model=model)
    return glm


def contrast_name(definition, contrast_names=None):
    if contrast_names is None:
        contrast_names = {}
    conds = sorted(definition.keys())
    rep = lambda c: contrast_names[c] if c in contrast_names else c
    pos = '_'.join([rep(c) for c in conds if definition[c] > 0])
    neg = '_'.join([rep(c) for c in conds if definition[c] < 0])
    return '%s-%s' % (pos, neg) if neg else pos


def estimate(spmmat_or_doc, contrast_definitions=None,
             out_dir=tempfile.gettempdir(), model='ar1',
             create_snapshots=True, keep_doc=True,
             contrast_names=None, mem=None, **options):

    _fit_glm = mem.cache(fit_glm) if mem else fit_glm

    if isinstance(spmmat_or_doc, dict):
        doc = spmmat_or_doc
    else:
        doc = load_intra(spmmat_or_doc, inputs=True, outputs=True, **options)

    if contrast_definitions is None:
        contrast_definitions = [{k: 1} for k in doc['contrasts']]
    if contrast_names is None:
        contrast_names = {}

    print 'Estimating %s %s' % (doc['study'], doc['subject'])

    # write directory
    write_dir = os.path.join(out_dir, doc['study'], doc['subject'])
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    if keep_doc:
        json.dump(doc, open(os.path.join(write_dir, 'doc.json'), 'wb'))

    # timing
    n_scans = doc['n_scans']
    n_runs = len(n_scans)

    # design matrix
    conds_ind = []
    vols_ind = []
    design_matrices = []
    spm_dm = np.array(doc['design_matrix'])[:, :-n_runs]

    pl.clf()
    pl.matshow(spm_dm)
    pl.savefig(os.path.join(write_dir, 'design_matrix_SPM.png'))

    i = 0
    for run, s_scans in enumerate(n_scans):
        dm = spm_dm[i:i + s_scans, :]
        run_ind, = np.where(np.logical_not(np.sum(dm, axis=0) == 0))
        conds_ind.append(run_ind)
        vols_ind.append((i, i + s_scans))
        design_matrices.append(dm[:, run_ind])
        i += s_scans
        if create_snapshots:
            pl.clf()
            pl.matshow(dm[:, run_ind])
            pl.savefig(os.path.join(
                write_dir, 'design_matrix_run#%s.png' % run))

    contrasts = {}
    for run_ind in conds_ind:                # cond indices per run
        for condef in contrast_definitions:
            name = contrast_name(condef, contrast_names)
            conds = np.zeros(len(run_ind))
            for k in condef:
                conds += np.array(doc['contrasts'][k])[run_ind] * condef[k]
            if np.unique(conds).size != 1:
                conds /= float(conds.max())
            contrasts.setdefault(name, []).append(conds)

    data = get_frmi_runs(doc['data'], vols_ind)

    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print '  Contrast % 2i out of %i: %s' % (
            index + 1, len(contrasts), contrast_id)

        z_image_path = os.path.join(write_dir, '%s_z_map.nii.gz' % contrast_id)
        con_runs = np.array([np.unique(c).size != 1 for c in contrast_val])

        d = np.array(data)[con_runs].tolist()
        c = np.array(contrast_val)[con_runs].tolist()
        m = [dm for run, dm in zip(con_runs, design_matrices) if run]
        glm = _fit_glm(d, m, doc['mask'], True, model=model)

        z_map, = glm.contrast(c, con_id=contrast_id, output_z=True)
        nb.save(z_map, z_image_path)

        # Create snapshots of the contrasts
        if create_snapshots:
            pl.clf()
            z_map_data = np.array(z_map.get_data(), copy=True)
            vmax = max(- z_map_data.min(), z_map_data.max())
            plot_map(z_map_data, z_map.get_affine(),
                     cmap=cm.cold_hot,
                     vmin=- vmax,
                     vmax=vmax,
                     anat=None,
                     figure=10,
                threshold=2.5)
            pl.savefig(os.path.join(
                write_dir, '%s_z_map.png' % contrast_id))
