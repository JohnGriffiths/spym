import re
import os
import glob


def get_contrast_names(level1_task_dir):
    with open(os.path.join(level1_task_dir, 'design.con'), 'rb') as f:
        contrasts = dict([
            re.findall('\/ContrastName(\d+)\s(.*)\s', l)[0]
            for l in f.read().split('\n') if l.startswith('/Contrast')])
    if contrasts == {} and os.path.join()

    return contrasts


def get_subject_stats(level2_task_dir, dtype='t', names=None):
    stat_files = os.path.join(level2_task_dir, 'cope*.feat',
                              'stats', '%sstat1.nii.gz' % dtype)
    stats = {}
    for stat_file in glob.glob(stat_files):
        stat_id = re.findall('cope(\d+).feat',
                             stat_file.split(os.path.sep)[-3])[0]

        if names is not None:
            stat_id = names[stat_id]
        stats.setdefault(stat_id, stat_file)

    return stats


def get_study_stats(study_dir):
    subjects = os.path.join(study_dir, 'sub???')
    contrasts = {}
    for subject_dir in glob.glob(subjects):
        subject_id = os.path.split(subject_dir)[1]
        level1_task_dirs = os.path.join(
            subject_dir, 'model', 'model001', 'task???_run???.feat')
        level2_task_dirs = os.path.join(
            subject_dir, 'model', 'model001', 'task???.gfeat')
        tasks_contrasts_names = {}
        for lvl1_dir in glob.glob(level1_task_dirs):
            task_id = os.path.split(lvl1_dir)[1].split('_run')[0]
            tasks_contrasts_names.setdefault(task_id, {}).update(
                get_contrast_names(lvl1_dir))
        for lvl2_dir in glob.glob(level2_task_dirs):
            task_id = os.path.split(lvl2_dir)[1].split('.gfeat')[0]
            stat_files = get_subject_stats(
                lvl2_dir, names=tasks_contrasts_names[task_id])
            contrasts.setdefault(subject_id, {}).update(stat_files)

    return contrasts
