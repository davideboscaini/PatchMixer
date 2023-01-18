import sys
sys.path.append('..')

import argparse
import numpy as np
import os
import shutil
from utils import get_logger, sorted_alphanumeric


def main():

    methods = ['pn', 'pn2', 'dgcnn', 'ours']

    path_weights = os.path.join('..', 'data', 'exps', 'weights')

    # Remove incomplete trainings
    # exps = sorted_alphanumeric(os.listdir(path_weights))
    # for exp in exps:
    #     if not os.path.isfile(os.path.join(path_weights, exp, 'last.pth')):
    #         shutil.rmtree(os.path.join(path_weights, exp))

    path_file = os.path.join(path_weights, 'summary_grasp.txt')

    with open(path_file, 'w') as f:

        exps = sorted_alphanumeric(os.listdir(path_weights))

        for test in ['rk10', 'rr10']:

            f.write('Test on: {:s}\n'.format(test))
            f.write('{:20s}: rk10   rr10   sk10   sr10   AvgTL\n'.format('Method'))

            processed_exps = list()

            for exp in exps:

                if len(exp.split('_')) == 4 and exp.split('_')[2] in methods:

                    if len(exp.split('_')) == 4:

                        print('[i] Processing exp:', exp)
                        exp_name = '{:s}_{:s}'.format(exp.split('_')[2], exp.split('_')[3])

                        oa = dict()
                        count = 0

                        for train in ['rk10', 'rr10', 'sk10', 'sr10']:

                            path_log = os.path.join(path_weights, 'grasp_{:s}_{:s}'.format(train, exp_name))

                            if os.path.exists(path_log) and path_log not in processed_exps:

                                fn = 'log_test_{:s}_last.txt'.format(test)
                                with open(os.path.join(path_log, fn)) as _:
                                    lines = [line for line in _.readlines()]
                                    last = lines[-1]
                                    oa[train] = last.split(' ')[5][:-1]
                                count += 1

                                processed_exps.append(path_log)

                        if count == 4:
                            if test == 'rk10':
                                moa = (float(oa['rr10']) + float(oa['sk10']) + float(oa['sr10'])) / 3
                            elif test == 'rr10':
                                moa = (float(oa['rk10']) + float(oa['sk10']) + float(oa['sr10'])) / 3
                            f.write('{:20s}: {:6s} {:6s} {:6s} {:6s} {:.4f}\n'.format(
                                exp_name, oa['rk10'], oa['rr10'], oa['sk10'], oa['sr10'], moa))

            f.write('\n')


if __name__ == '__main__':
    main()
