import matplotlib
matplotlib.use('Agg')

import os
import glob

import nibabel as nb
import numpy as np
import pylab as pl

from scipy import stats
from nipy.labs.viz import plot_map
from nipy.labs.viz_tools import cm
from joblib import Parallel, delayed

from parsing_openfmri import load_openfmri

# GLOBALS
pwd = os.path.dirname(os.path.abspath(__file__))

mask_file = os.path.join(pwd, 'mask.nii.gz')
mask = nb.load(mask_file).get_data().astype('bool')
affine = nb.load(mask_file).get_affine()


def plot_study_models(study_dir, out_dir, model_id,
                      hrf_model='canonical with derivative',
                      drift_model='cosine',
                      n_jobs=-1, verbose=1):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    docs = load_openfmri(study_dir, model_id,
                         hrf_model=hrf_model,
                         drift_model=drift_model,
                         n_jobs=n_jobs, verbose=verbose)

    Parallel(n_jobs=n_jobs)(
        delayed(_plot_study_models)(doc, out_dir, model_id) for doc in docs)


def _plot_study_models(doc, out_dir, model_id):
        for session_id, dm in zip(doc['sessions_id'],
                                  doc['design_matrices_object']):

            dm.show()
            fname = '_'.join([doc['study_id'], doc['subject_id'],
                              model_id, session_id])
            pl.savefig(os.path.join(out_dir, '%s.png' % fname), pdi=500)


def plot_study_maps(study_dir, out_dir, model_id=None,
                    dtype='t', contrasts=None, n_jobs=-1):
    try:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    except:
        pass

    group_maps = {}

    study_id = os.path.split(study_dir)[1]

    if contrasts is not None:
        for x in glob.iglob(os.path.join(study_dir, contrasts)):
            map_id = x.split(os.path.sep)[-1].split('.nii.gz')[0]
            model_id = x.split(os.path.sep)[-3]
            group_id = '%s_%s_%s' % (study_id, model_id, map_id)
            data = nb.load(x).get_data()[mask]
            group_maps.setdefault(group_id, []).append(data)

    else:
        for subject_dir in glob.iglob(os.path.join(study_dir, 'sub???')):
            if model_id is None:
                models = os.path.join(subject_dir, 'model', '*')
            else:
                models = os.path.join(subject_dir, 'model', model_id)

            for model_dir in glob.iglob(models):
                model_id = os.path.split(model_dir)[1]
                maps_dir = os.path.join(model_dir, '%s_maps' % dtype)
                for x in glob.iglob(os.path.join(maps_dir, '*.nii.gz')):
                    map_id = os.path.split(x)[1].split('.nii.gz')[0]
                    group_id = '%s_%s_%s' % (study_id, model_id, map_id)
                    data = nb.load(x).get_data()[mask]
                    group_maps.setdefault(group_id, []).append(data)

    Parallel(n_jobs=n_jobs)(
        delayed(_plot_group_map)(
            k, group_maps[k], out_dir) for k in group_maps)


def _plot_group_map(label, individual_maps, out_dir):
    print 'outputting %s' % label
    mean_map = np.zeros(mask.shape)
    mean_map[mask] = np.mean(individual_maps, axis=0)
    vmax = np.abs(mean_map).max()
    threshold = stats.scoreatpercentile(mean_map[mask], 95)
    map_id = label.split('_', 2)[-1]
    study_id = label.split('_', 2)[-3]

    title = '%s__%s' % (study_id, map_id)
    plot_map(mean_map, affine, slicer='z',
             cut_coords=7,
             vmin=-vmax, vmax=vmax, threshold=threshold,
             cmap=cm.cold_hot, title=title)
    pl.savefig(os.path.join(out_dir, '%s.png' % title), dpi=200)
    img = nb.Nifti1Image(mean_map, affine=affine)
    nb.save(img, os.path.join(out_dir, '%s.nii.gz' % title))


if __name__ == '__main__':
    from optparse import OptionParser
    from utils import print_command_line_options

    parser = OptionParser()
    parser.add_option("-d", "--data-dir", dest="study_dir",
                      help="full path to study directory")
    parser.add_option("-r", "--report", choices=('maps', 'models'),
                      help="the type of reporting to do",
                      dest='report')
    parser.add_option("-m", "--model-id",
                      help="model id. Mandatory for models report",
                      dest='model_id')
    parser.add_option("-c", "--contrasts",
                      help=("The pattern of the maps to plot, "
                            "relative to data_dir. Works only with "
                            "--report=maps. In this case "
                            "model_id and dtype are ignored"),
                      dest='contrasts')
    parser.add_option("-o", "--out-dir",
                      help="output directory. Created if does not exists.",
                      dest='out_dir')
    parser.add_option(
        "-t", "--dtype",
        help="Type of stat to report. Defaults to 't'. Only for maps report",
        default='t',
        choices=('t', 'c', 'z', 'var'), dest='dtype')
    parser.add_option("-n", "--n-jobs",
                      help="number of parallel jobs. -1 means all cores.",
                      dest='n_jobs', type='int', default=-1)
    parser.add_option("--hrf-model",
                      dest='hrf_model', default='canonical with derivative',
                      help="hrf model for design matrix")
    parser.add_option("--drift-model",
                      dest='drift_model', default='cosine',
                      help='hrf model for design matrix')

    (options, args) = parser.parse_args()
    print_command_line_options(options)

    if options.report == 'maps':
        plot_study_maps(study_dir=options.study_dir,
                        out_dir=options.out_dir,
                        model_id=options.model_id,
                        dtype=options.dtype,
                        contrasts=options.contrasts,
                        n_jobs=options.n_jobs)
    elif options.report == 'models':
        plot_study_models(study_dir=options.study_dir,
                          out_dir=options.out_dir,
                          model_id=options.model_id,
                          hrf_model=options.hrf_model,
                          drift_model=options.drift_model,
                          n_jobs=options.n_jobs)
