import matplotlib
matplotlib.use('Agg')

import os
import glob

import nibabel as nb
import numpy as np
import pylab as pl

from nipy.labs.viz import plot_map
from nipy.labs.viz_tools import cm
from joblib import Parallel, delayed

from parse_openfmri import load_openfmri

# GLOBALS
pwd = os.path.dirname(os.path.abspath(__file__))

mask_file = os.path.join(pwd, 'mask.nii.gz')
mask = nb.load(mask_file).get_data().astype('bool')
affine = nb.load(mask_file).get_affine()


def plot_study_models(study_dir, model_id, out_dir,
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
        delayed(_plot_study_models)(doc, model_id, out_dir) for doc in docs)


def _plot_study_models(doc, model_id, out_dir):
        for session_id, dm in zip(doc['sessions_id'],
                                  doc['design_matrices_object']):

            dm.show()
            fname = '_'.join([doc['study_id'], doc['subject_id'],
                              model_id, session_id])
            pl.savefig(os.path.join(out_dir, 'design_%s.png' % fname), pdi=500)


def plot_study_maps(study_dir, out_dir, model_id=None, dtype='t', n_jobs=-1):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    group_maps = {}

    study_id = os.path.split(study_dir)[1]
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
    plot_map(mean_map, affine, slicer='z',
             cut_coords=[-40, -20, -5, 0, 10, 30, 60],
             vmin=-vmax, vmax=vmax, threshold=1.,
             cmap=cm.cold_hot, title=label)
    pl.savefig(os.path.join(out_dir, 'group_%s.png' % label), dpi=500)


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-d", "--data-dir", dest="study_dir",
                      help="Full path to study directory")
    parser.add_option("-r", "--report", choices=('maps', 'models'),
                      help="The type of reporting to do",
                      dest='report')
    parser.add_option("-m", "--model-id",
                      help="Model id. Mandatory for models report",
                      dest='model_id')
    parser.add_option("-o", "--out-dir",
                      help="Output directory. Created if does not exists.",
                      dest='out_dir')
    parser.add_option(
        "-t", "--dtype",
        help="Type of stat to report. Defaults to 't'. Only for maps report",
        default='t',
        choices=('t', 'c', 'z', 'var'), dest='dtype')
    parser.add_option("-n", "--n-jobs",
                      help="Number of parallel jobs. -1 means all cores.",
                      dest='n_jobs', type='int', default=-1)
    parser.add_option("--hrf-model",
                      dest='hrf_model', default='canonical with derivative')
    parser.add_option("--drift-model",
                      dest='drift_model', default='cosine')

    (options, args) = parser.parse_args()

    if options.report == 'maps':
        plot_study_maps(options.study_dir, options.out_dir,
                        options.model_id, options.dtype,
                        options.n_jobs)
    elif options.report == 'models':
        plot_study_models(options.study_dir, options.model_id,
                          options.out_dir,
                          options.hrf_model,
                          options.drift_model,
                          options.n_jobs)
