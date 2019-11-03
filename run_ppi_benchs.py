#!/usr/bin/env python
"""
Usage:
    run_ppi_benchs.py [options] LOG_TARGET_DIR

Options:
    -h --help         Show this screen.
    --num-runs NUM    Number of runs to perform for each configuration. [default: 10]
    --debug           Turn on debugger.
"""
import os
import subprocess
import re
import numpy as np

from docopt import docopt
from dpu_utils.utils import run_and_debug

MODEL_TYPES = ["GGNN", "RGCN", "RGAT", "RGIN", "GNN-Edge-MLP0", "GNN-Edge-MLP1", "GNN_FiLM"]

TEST_RES_RE = re.compile('^Metrics: Avg MicroF1: (0.\d+)')
TIME_RE = re.compile('^Training took (\d+)s')


def run(args):
    target_dir = args['LOG_TARGET_DIR']
    os.makedirs(target_dir, exist_ok=True)
    print("Starting PPI experiments, will write logfiles for runs into %s." % target_dir)
    num_seeds = int(args.get('--num-runs'))
    print("| %- 13s | %- 17s | %- 10s |" % ("Model", "Avg. MicroF1", "Avg. Time"))
    print("|" + "-" * 15 + "|" + "-" * 19 + "|" + "-" * 12 + "|")
    for model in MODEL_TYPES:
        model_f1s = []
        model_times = []
        for seed in range(1, 1 + num_seeds):
            logfile = os.path.join(target_dir, "%s_seed%i.txt" % (model.lower(), seed))
            with open(logfile, "w") as log_fh:
                subprocess.check_call(["python",
                                       "train.py",
                                       "--quiet",
                                       "--run-test",
                                       model,
                                       "PPI",
                                       "--model-param-overrides",
                                       "{\"random_seed\": %i}" % seed,
                                       ],
                                      stdout=log_fh,
                                      stderr=log_fh)
            with open(logfile, "r") as log_fh:
                for line in log_fh.readlines():
                    time_match = TIME_RE.search(line)
                    res_match = TEST_RES_RE.search(line)
                    if time_match is not None:
                        model_times.append(int(time_match.groups()[0]))
                    elif res_match is not None:
                        model_f1s.append(float(res_match.groups()[0]))

        print("| %- 13s | %.3f (+/- %.3f) |     % 4.1f |"
              % (model,
                 np.mean(model_f1s),
                 np.std(model_f1s),
                 np.mean(model_times)))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])
