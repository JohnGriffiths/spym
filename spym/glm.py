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
