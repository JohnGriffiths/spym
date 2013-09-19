import os
import re
import gzip

import numpy as np
import nibabel as nb
import scipy.io as sio


class Niimg(object):

    def __init__(self, path):
        img = nb.load(path)
        self._path = path
        self.shape = img.shape
        self._affine = img.get_affine()

    def get_affine(self):
        return self._affine

    def get_data(self):
        img = nb.load(self._path)
        return img.get_data()


def fix_experiment(docs, fix=None, fields=None):
    if isinstance(docs, dict):
        docs = [docs]
    if fields is None:
        fields = ['t_maps', 'c_maps', 'task_contrasts', 'design_contrasts']
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

        fixed_conditions = []
        for session_conditions in doc['condition_key']:
            fixed_cond = []
            for cond in session_conditions:
                if cond in fix:
                    fixed_cond.append(fix[cond])
                else:
                    fixed_cond.append(cond)
            fixed_conditions.append(fixed_cond)
        fixed_doc['condition_key'] = fixed_conditions

        fixed_docs.append(fixed_doc)

    return fixed_docs


def fix_subject_ids(docs, fix):
    subjects = sorted([doc['subject_id'] for doc in docs])
    norm = dict([(sid, 'sub%03i' % (i + 1))
                for i, sid in enumerate(subjects)])
    mapping = {}
    for doc in docs:
        mapping[norm[doc['subject_id']]] = fix[doc['subject_id']]['subject_id']
        doc['subject_id'] = norm[doc['subject_id']]
    return docs, mapping


def load_matfile(location):
    if location.endswith('.gz'):
        return sio.loadmat(
            gzip.open(location, 'rb'),
            squeeze_me=True,
            struct_as_record=False
            )

    return sio.loadmat(
        location, squeeze_me=True, struct_as_record=False)


def find_data_dir(wd, fpath):
    fpath = fpath.strip()

    def right_splits(p):
        yield p
        while p not in ['', None]:
            p = p.rsplit(os.path.sep, 1)[0]
            yield p

    def left_splits(p):
        yield p
        while len(p.split(os.path.sep, 1)) > 1:
            p = p.split(os.path.sep, 1)[1]
            yield p

    if not os.path.isfile(fpath):
        for rs in right_splits(wd):
            if not os.path.exists(rs):
                continue
            for ls in left_splits(fpath):
                p = os.path.join(rs, *ls.split(os.path.sep))
                if os.path.isfile(p):
                    return os.path.dirname(p)
    else:
        return os.path.dirname(fpath)
    raise Exception('bla')
    return ''


def prefix_filename(path, prefix):
    path, filename = os.path.split(str(path))
    return os.path.join(path, '%s%s' % (prefix, filename))


def strip_prefix_filename(path, len_strip):
    path, filename = os.path.split(str(path))
    return os.path.join(path, filename[len_strip:])


def remove_special(name):
    return re.sub("[^0-9a-zA-Z\-]+", '_', name)


def check_niimgs(niimgs):
    niimgs_ = []
    for niimg in niimgs:
        if isinstance(niimg, (str, unicode)):
            niimgs_.append(Niimg(niimg))
        elif isinstance(niimg, list):
            niimgs_.append(nb.concat_images(niimg))
        else:
            niimgs_.append(niimg)
    return niimgs_


def check_design_matrices(design_matrices):
    return [np.array(dm) for dm in design_matrices]


def check_contrasts(contrasts):
    checked = {}

    for k in contrasts:
        for session_contrast in contrasts[k]:
            session_contrast = np.array(session_contrast)
            if not len(session_contrast.shape) > 1:
                checked.setdefault(k, []).append(session_contrast)
            else:
                break

    return checked


def report(preproc_docs=None, intra_docs=None):
    if preproc_docs is not None:
        doc = preproc_docs[0]

        print '#' * 80
        print 'preproc'
        print '#' * 80

        print
        print '-' * 80
        print 'slice timing'
        print '-' * 80

        try:
            print '  bold', all([os.path.exists(p)
                                 for s in doc['slice_timing']['bold'] for p in s])
            print '  n_slices', doc['slice_timing']['n_slices']
            print '  ref_slice', doc['slice_timing']['ref_slice']
            print '  slices_order', doc['slice_timing']['slices_order']
            print '  ta', doc['slice_timing']['ta']
            print '  tr', doc['slice_timing']['tr']
        except:
            pass

        print
        print '-' * 80
        print 'realign'
        print '-' * 80

        try:
            print '  bold', all([os.path.exists(p)
                                 for s in doc['realign']['bold'] for p in s])
            print '  motion', all([os.path.exists(p)
                                   for p in doc['realign']['motion']])

        except:
            pass

        print
        print '-' * 80
        print 'preproc/normalize'
        print '-' * 80
        try:

            print '  anatomy', os.path.exists(doc['preproc']['anatomy'])
            print '  norm anatomy', os.path.exists(doc['normalize']['anatomy'])
            print '  norm mat_file', os.path.exists(
                doc['normalize']['mat_file'])
            print '  bold', all([
                os.path.exists(p)
                for s in doc['normalize']['bold'] for p in s])
        except:
            pass

        print
        print '-' * 80
        print 'coregistration'
        print '-' * 80

        try:
            print '  realigned', all([
                os.path.exists(p) for p in doc['coregistration']['realigned']])
            print '  anatomy', os.path.exists(doc['coregistration']['anatomy'])
            print '  realigned_ref', os.path.exists(
                doc['coregistration']['realigned_ref'])
        except:
            pass

        print
        print '-' * 80
        print 'smooth'
        print '-' * 80

        try:
            print '  fwhm', doc['smooth']['fwhm']
            print '  bold', all([
                os.path.exists(p)
                for s in doc['smooth']['bold'] for p in s])
        except:
            pass

        print
        print '-' * 80
        print 'final'
        print '-' * 80

        try:

            print '  anatomy', os.path.exists(doc['final']['anatomy'])
            print '  bold', all([
                os.path.exists(p)
                for s in doc['final']['bold'] for p in s])
        except:
            pass

    if intra_docs is not None:
        doc = intra_docs[0]

        print
        print '#' * 80
        print 'stats intra'
        print '#' * 80

        print
        print '-' * 80
        print 'experiment'
        print '-' * 80

        print '  n_scans', doc['n_scans']
        print '  n_sessions', doc['n_sessions']
        print '  tr', doc['tr']

        print
        print '-' * 80
        print 'maps'
        print '-' * 80

        print '  beta_maps', all([os.path.exists(p) for p in doc['beta_maps']])
        print '  c_maps', all([os.path.exists(doc['c_maps'][c])
                               for c in doc['c_maps']])

        print '  t_maps', all([os.path.exists(doc['t_maps'][c])
                               for c in doc['t_maps']])
        print '  mask', os.path.exists(doc['mask'])

        print
        print '-' * 80
        print 'experimental conditions'
        print '-' * 80
        for i, session_conditions in enumerate(doc['condition_key']):
            print '  session%03i:' % (i + 1)
            for cond in session_conditions:
                print '   ', cond
