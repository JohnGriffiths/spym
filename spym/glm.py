import os
from itertools import izip

import numpy as np
import nibabel as nb

from nipy.modalities.fmri.glm import FMRILinearModel
from joblib import Parallel, delayed

from parsing_openfmri import _load_openfmri
from utils import check_niimgs, check_design_matrices, check_contrasts
from utils import remove_special


def _first_level(out_dir, data, design_matrices, contrasts,
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


def openfmri_first_level(study_dir, subjects_id, model_id,
                         hrf_model='canonical with derivative',
                         drift_model='cosine',
                         glm_model='ar1', n_jobs=-1, verbose=1):
    """ Utility function to compute first level GLMs in parallel
    """

    if n_jobs == 1:
        for subject_id in subjects_id:
            _openfmri_first_level(study_dir, subject_id, model_id,
                                  hrf_model, drift_model, glm_model,
                                  verbose - 1)
    else:
        Parallel(n_jobs=n_jobs)(delayed(_openfmri_first_level)(
            study_dir, subject_id, model_id,
            hrf_model, drift_model, glm_model,
            verbose - 1) for subject_id in subjects_id)


def _openfmri_first_level(study_dir, subject_id, model_id,
                          hrf_model='canonical with derivative',
                          drift_model='cosine',
                          glm_model='ar1', verbose=1):
    study_id = os.path.split(study_dir)[1]

    if verbose > 0:
        print '%s@%s: first level glm' % (subject_id, study_id)

    doc = _load_openfmri(study_dir, subject_id, model_id,
                         hrf_model, drift_model, verbose)

    model_dir = os.path.join(study_dir, subject_id, 'model', model_id)

    _first_level(model_dir, doc['data'], doc['design_matrices'],
                 doc['task_contrasts'], glm_model,
                 mask='compute', verbose=verbose - 1)

if __name__ == '__main__':
    from optparse import OptionParser

    from parsing_openfmri import get_subjects_id
    from utils import print_command_line_options

    parser = OptionParser()
    parser.add_option("-d", "--data-dir", dest="study_dir",
                      help="Full path to study directory")
    parser.add_option("-m", "--model-id",
                      help="Model id. Mandatory for models report",
                      dest='model_id')
    parser.add_option("-n", "--n-jobs",
                      help="number of parallel jobs. -1 means all cores.",
                      dest='n_jobs', type='int', default=-1)
    parser.add_option("-s", "--subject-id",
                      help=("specify a subject_id to process "
                            "only this one. Default processes all"),
                      dest='subject_id')
    parser.add_option("--hrf-model",
                      dest='hrf_model', default='canonical with derivative',
                      help="hrf model for design matrix")
    parser.add_option("--drift-model",
                      dest='drift_model', default='cosine',
                      help='hrf model for design matrix')
    parser.add_option("--glm-model",
                      dest='glm_model', default='ar1',
                      help='glm model')

    (options, args) = parser.parse_args()

    if options.subject_id is not None:
        options.subject_id = [options.subject_id]
    else:
        options.subject_id = get_subjects_id(options.study_dir)

    print_command_line_options(options)

    openfmri_first_level(study_dir=options.study_dir,
                         subjects_id=options.subject_id,
                         model_id=options.model_id,
                         hrf_model=options.hrf_model,
                         drift_model=options.drift_model,
                         glm_model=options.glm_model,
                         n_jobs=options.n_jobs,
                         verbose=1)
