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


def estimate(spmmat, contrast_definitions=None,
             out_dir=tempfile.gettempdir(), model='ar1',
             create_snapshots=True, keep_doc=True,
             contrast_names=None, **options):

    doc = load_intra(spmmat, inputs=True, outputs=True, **options)
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
    n_sessions = len(n_scans)

    # design matrix
    cond_idx = []
    time_idx = []
    design_matrices = []
    spm_dm = np.array(doc['design_matrix'])[:, :-n_sessions]

    pl.clf()
    pl.matshow(spm_dm)
    pl.savefig(os.path.join(write_dir, 'design_matrix_SPM.png'))

    i = 0
    for session, s_scans in enumerate(n_scans):
        dm = spm_dm[i:i + s_scans, :]
        session_idx, = np.where(np.logical_not(np.sum(dm, axis=0) == 0))
        cond_idx.append(session_idx)
        time_idx.append((i, i + s_scans))
        design_matrices.append(dm[:, session_idx])
        i += s_scans
        if create_snapshots:
            pl.clf()
            pl.matshow(dm[:, session_idx])
            pl.savefig(os.path.join(
                write_dir, 'design_matrix_session#%s.png' % session))

    sessions_contrasts = []
    for session in range(n_sessions):
        c = {}
        for def_ in contrast_definitions:
            positive = '_'.join(
                [k if k not in contrast_names else contrast_names[k]
                 for k in sorted(def_.keys()) if def_[k] > 0])
            negative = '_'.join(
                [k if k not in contrast_names else contrast_names[k]
                 for k in sorted(def_.keys()) if def_[k] < 0])
            if negative != '':
                name = '%s-%s' % (positive, negative)
            else:
                name = positive
            conditions = np.zeros(len(cond_idx[session]))
            for k in def_:
                conditions += np.array(
                    doc['contrasts'][k])[cond_idx[session]] * def_[k]

            c[name] = conditions / conditions.max()
        sessions_contrasts.append(c)

    for j, (design_matrix, contrasts, time_indices) in enumerate(zip(
            design_matrices, sessions_contrasts, time_idx)):

        data = get_fmri_data(doc['data'], time_indices)
        fmri_glm = FMRILinearModel(data, design_matrix, mask=doc['mask'])
        fmri_glm.fit(do_scaling=True, model=model)

        # estimate the contrasts
        for i, (contrast_id, contrast_val) in enumerate(contrasts.iteritems()):

            # save the z_image
            image_path = os.path.join(
                write_dir, 'z_map#%s_session#%s.nii.gz' % (contrast_id, j))
            z_map, = fmri_glm.contrast(
                contrast_val, con_id=contrast_id, output_z=True)
            nb.save(z_map, image_path)

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
                    write_dir, 'z_map#%s_session#%s.png' % (contrast_id, j)))
