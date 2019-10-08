#!/usr/bin/env python
"""
Usage:
   train.py [options] MODEL_NAME TASK_NAME

MODEL_NAME has to be one of the supported models, which currently are
 GGNN, GNN-Edge-MLP, GNN-FiLM, RGAT, RGCN, RGDCN

Options:
    -h --help                       Show this screen.
    --data-path PATH                Path to load data from, has task-specific defaults under data/.
    --result-dir DIR                Directory to store logfiles and trained models. [default: trained_models]
    --run-test                      Indicate if the task's test should be run.
    --model-param-overrides PARAMS  Parameter settings overriding model defaults (in JSON format).
    --task-param-overrides PARAMS   Parameter settings overriding task defaults (in JSON format).
    --quiet                         Show less output.
    --tensorboard DIR               Dump tensorboard event files to DIR.
    --azure-info=<path>             Azure authentication information file (JSON). [default: azure_auth.json]
    --debug                         Turn on debugger.
"""
import json
import os
import sys
import time

from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath, git_tag_run

from utils.model_utils import name_to_model_class, name_to_task_class
from test import test


def run(args):
    azure_info_path = args.get('--azure-info', None)
    model_cls, additional_model_params = name_to_model_class(args['MODEL_NAME'])
    task_cls, additional_task_params = name_to_task_class(args['TASK_NAME'])

    # Collect parameters from first the class defaults, potential task defaults, and then CLI:
    task_params = task_cls.default_params()
    task_params.update(additional_task_params)
    model_params = model_cls.default_params()
    model_params.update(additional_model_params)

    # Load potential task-specific defaults:
    task_model_default_hypers_file = \
        os.path.join(os.path.dirname(__file__),
                     "tasks",
                     "default_hypers",
                     "%s_%s.json" % (task_cls.name(), model_cls.name(model_params)))
    if os.path.exists(task_model_default_hypers_file):
        print("Loading task/model-specific default parameters from %s." % task_model_default_hypers_file)
        with open(task_model_default_hypers_file, "rt") as f:
            default_task_model_hypers = json.load(f)
        task_params.update(default_task_model_hypers['task_params'])
        model_params.update(default_task_model_hypers['model_params'])

    # Load overrides from command line:
    task_params.update(json.loads(args.get('--task-param-overrides') or '{}'))
    model_params.update(json.loads(args.get('--model-param-overrides') or '{}'))

    # Finally, upgrade every parameters that's a path to a RichPath:
    task_params_orig = dict(task_params)
    for (param_name, param_value) in task_params.items():
        if param_name.endswith("_path"):
            task_params[param_name] = RichPath.create(param_value, azure_info_path)

    # Now prepare to actually run by setting up directories, creating object instances and running:
    result_dir = args.get('--result-dir', 'trained_models')
    os.makedirs(result_dir, exist_ok=True)
    task = task_cls(task_params)
    data_path = args.get('--data-path') or task.default_data_path()
    data_path = RichPath.create(data_path, azure_info_path)
    task.load_data(data_path)

    random_seeds = model_params['random_seed']
    if not isinstance(random_seeds, list):
        random_seeds = [random_seeds]

    for random_seed in random_seeds:
        model_params['random_seed'] = random_seed
        run_id = "_".join([task_cls.name(), model_cls.name(model_params), time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])

        model = model_cls(model_params, task, run_id, result_dir)
        model.log_line("Run %s starting." % run_id)
        model.log_line(" Using the following task params: %s" % json.dumps(task_params_orig))
        model.log_line(" Using the following model params: %s" % json.dumps(model_params))

        if sys.stdin.isatty():
            try:
                git_sha = git_tag_run(run_id)
                model.log_line(" git tagged as %s" % git_sha)
            except:
                print(" Tried tagging run in git, but failed.")
                pass

        model.initialize_model()
        model.train(quiet=args.get('--quiet'), tf_summary_path=args.get('--tensorboard'))

        if args.get('--run-test'):
            test(model.best_model_file, data_path, result_dir, quiet=args.get('--quiet'), run_id=run_id)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])
