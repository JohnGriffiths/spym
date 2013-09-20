import re
import os
import glob


def get_contrast_names(level1_task_dir):
    with os.path.join(level1_task_dir, 'design.con') as f:
        contrasts = dict([
            re.findall('\/ContrastName(\d+)\s(.*)\s', l)[0]
            for l in f.read().split('\n') if l.startswith('/Contrast')])
    return contrasts


def get_stats(level2_task_dir, dtype='t', names=None):
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
