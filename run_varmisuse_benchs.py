#!/usr/bin/env python
"""
Usage:
    run_varmisuse_benchs.py [options] LOG_TARGET_DIR

Options:
    -h --help         Show this screen.
    --num-runs NUM    Number of runs to perform for each configuration. [default: 5]
    --debug           Turn on debugger.
"""
import os
import subprocess
import re
import numpy as np

from docopt import docopt
from dpu_utils.utils import run_and_debug

MODEL_TYPES = ["GGNN", "RGCN", "RGAT", "RGIN", "GNN-Edge-MLP0", "GNN-Edge-MLP1", "GNN_FiLM"]

TEST_RES_RE = re.compile('^Metrics: Accuracy: (0.\d+)')
VALID_RES_RE = re.compile('Best validation results: Accuracy: (0.\d+)')
MODEL_FILE_RE = re.compile('^Loading model from file (.+)\.')


def run(args):
    target_dir = args['LOG_TARGET_DIR']
    os.makedirs(target_dir, exist_ok=True)
    print("Starting VarMisuse experiments, will write logfiles for runs into %s." % target_dir)
    num_seeds = int(args.get('--num-runs'))
    print("| %- 14s | %- 17s | %- 17s | %- 17s |" % ("Model",
                                                     "Valid Acc",
                                                     "Test Acc",
                                                     "TestOnly Acc"))
    print("|" + "-" * 16 + "|" + "-" * 19 + "|" + "-" * 19 + "|" + "-" * 19 + "|")
    for model in MODEL_TYPES:
        valid_accs, test_accs, testonly_accs = [], [], []
        for seed in range(1, 1 + num_seeds):
            logfile = os.path.join(target_dir, "%s_seed%i.txt" % (model.lower(), seed))
            test_logfile = os.path.join(target_dir, "%s_seed%i-testonly.txt" % (model.lower(), seed))
            with open(logfile, "w") as log_fh:
                subprocess.check_call(["python",
                                       "train.py",
                                       "--quiet",
                                       "--run-test",
                                       model,
                                       "VarMisuse",
                                       "--model-param-overrides",
                                       "{\"random_seed\": %i}" % seed,
                                       ],
                                      stdout=log_fh,
                                      stderr=log_fh)
            model_file = None
            with open(logfile, "r") as log_fh:
                for line in log_fh.readlines():
                    valid_res_match = VALID_RES_RE.search(line)
                    test_res_match = TEST_RES_RE.search(line)
                    model_file_match = MODEL_FILE_RE.search(line)
                    if valid_res_match is not None:
                        valid_accs.append(float(valid_res_match.groups()[0]))
                    elif test_res_match is not None:
                        test_accs.append(float(test_res_match.groups()[0]))
                    elif model_file_match is not None:
                        model_file = model_file_match.groups()[0]

            # Run TestOnly
            assert model_file is not None, "Could not find saved model file"
            with open(test_logfile, "w") as log_fh:
                subprocess.check_call(["python",
                                       "test.py",
                                       "--quiet",
                                       model_file,
                                       "data/varmisuse/graphs-testonly",
                                       ],
                                      stdout=log_fh,
                                      stderr=log_fh)
            with open(test_logfile, "r") as log_fh:
                for line in log_fh.readlines():
                    test_res_match = TEST_RES_RE.search(line)
                    if test_res_match is not None:
                        testonly_accs.append(float(test_res_match.groups()[0]))
 
        print("| %- 14s | %.3f (+/- %.3f) | %.3f (+/- %.3f) | %.3f (+/- %.3f) |"
              % (model,
                 np.mean(valid_accs),
                 np.std(valid_accs),
                 np.mean(test_accs),
                 np.std(test_accs),
                 np.mean(testonly_accs),
                 np.std(testonly_accs),
                ))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])
