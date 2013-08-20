import os

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.glm import FMRILinearModel
from joblib import Parallel, delayed

from utils import check_niimgs, check_design_matrices, check_contrasts
from utils import remove_special


def _first_level_glm(out_dir, data, design_matrices, contrasts,
                    glm_model='ar1', mask='compute', verbose=1):
    if verbose:
        print '%s:' % out_dir

    data = check_niimgs(data)
    design_matrices = check_design_matrices(design_matrices)
    contrasts = check_contrasts(contrasts)
    glm = FMRILinearModel(data, design_matrices, mask=mask)
    glm.fit(do_scaling=True, model=glm_model)

    for i, contrast_id in enumerate(contrasts):
        if verbose:
            print '  %s/%s - %s ' % (i, len(contrasts), contrast_id)

        con_val = []
        for session_con, session_dm in zip(contrasts[contrast_id],
                                           design_matrices):

            con = np.zeros(session_dm.shape[1])
            con[:len(session_con)] = session_con
            con_val.append(con)

        z_map, t_map, c_map, var_map = glm.contrast(
            con_val,
            con_id=contrast_id,
            output_z=True,
            output_stat=True,
            output_effects=True,
            output_variance=True,)

        for dtype, img in zip(['z', 't', 'c', 'var'],
                              [z_map, t_map, c_map, var_map]):

            map_dir = os.path.join(out_dir, '%s_maps' % dtype)

            if not os.path.exists(map_dir):
                os.makedirs(map_dir)

            path = os.path.join(
                map_dir, '%s.nii.gz' % remove_special(contrast_id))
            nb.save(img, path)

    nb.save(glm.mask, os.path.join(out_dir, 'mask.nii.gz'))


def first_level_glm(out_dirs, all_data, all_design_matrices,
                    contrasts, glm_model='ar1',
                    mask='compute', n_jobs=-1, verbose=1):
    """ Utility function to compute first level GLMs in parallel
    """

    Parallel(n_jobs=n_jobs)(delayed(
        _first_level_glm)(out_dir, data,
                          design_matrices, contrasts,
                          glm_model, mask, verbose)
        for out_dir, data, design_matrices in zip(
                out_dirs, all_data, all_design_matrices)
        )



# # ----------------------------------------------------------------------------
# # GLM on openfmri layout
# # ----------------------------------------------------------------------------


# def first_level_glm(study_dir, subjects_id, model_id,
#                      hrf_model='canonical with derivative',
#                      drift_model='cosine',
#                      glm_model='ar1', mask='compute', n_jobs=-1, verbose=1):
#     """ Utility function to compute first level GLMs in parallel
#     """

#     if n_jobs == 1:
#         for subject_id in subjects_id:
#             _openfmri_first_level_glm(study_dir, subject_id, model_id,
#                                       hrf_model, drift_model, glm_model,
#                                       mask, verbose - 1)
#     else:
#         Parallel(n_jobs=n_jobs)(delayed(
#             _openfmri_first_level_glm)(
#                 study_dir, subject_id, model_id,
#                 hrf_model, drift_model, glm_model, mask, verbose - 1)
#                 for subject_id in subjects_id
#             )


# def _openfmri_first_level_glm(study_dir, subject_id, model_id,
#                               hrf_model='canonical with derivative',
#                               drift_model='cosine',
#                               glm_model='ar1', mask='compute', verbose=1):
#     study_id = os.path.split(study_dir)[1]
#     subject_dir = os.path.join(study_dir, subject_id)

#     if verbose > 0:
#         print '%s@%s: first level glm' % (subject_id, study_id)

#     tr = get_scan_key(study_dir)['TR']
#     images = get_scans(subject_dir)
#     n_scans = get_scans_count(subject_dir)
#     motion = get_motion(subject_dir)
#     contrasts = get_task_contrasts(study_dir, subject_dir, model_id, hrf_model)
#     events = get_events(subject_dir)

#     design_matrices = make_design_matrices(events, n_scans, tr,
#                                            hrf_model, drift_model, motion)

#     import pylab as pl

#     for session_id, dm in enumerate(design_matrices):
#         pl.matshow(dm)
#         pl.savefig('/tmp/%s_%s.png' % (subject_id, session_id + 1), dpi=300)

#     model_dir = os.path.join(study_dir, subject_id, 'model', model_id)

#     _first_level_glm(model_dir, images, design_matrices,
#                      contrasts, glm_model, mask, verbose - 1)
