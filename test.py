#!/usr/bin/env python
"""
Usage:
   test.py [options] STORED_MODEL_PATH [DATA_PATH]

STORED_MODEL is the path of a model snapshot created by train.py.
DATA_PATH is the location of the data to test on.

Options:
    -h --help                       Show this screen.
    --result-dir DIR                Directory to store logfiles and trained models. [default: trained_models]
    --azure-info PATH               Azure authentication information file (JSON). [default: azure_auth.json]
    --quiet                         Show less output.
    --debug                         Turn on debugger.
"""
import json
from typing import Optional

from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath

from utils.model_utils import restore


def test(model_path: str, test_data_path: Optional[RichPath], result_dir: str, quiet: bool = False, run_id: str = None):
    model = restore(model_path, result_dir, run_id)
    model.params['max_nodes_in_batch'] = 2 * model.params['max_nodes_in_batch']  # We can process larger batches if we don't do training
    test_data_path = test_data_path or RichPath.create(model.task.default_data_path())
    model.log_line(" Using the following task params: %s" % json.dumps(model.task.params))
    model.log_line(" Using the following model params: %s" % json.dumps(model.params))
    model.test(test_data_path)


def run(args):
    azure_info_path = args.get('--azure-info', None)
    model_path = args['STORED_MODEL_PATH']
    test_data_path = args.get('DATA_PATH')
    if test_data_path is not None:
        test_data_path = RichPath.create(test_data_path, azure_info_path)
    result_dir = args.get('--result-dir', 'trained_models')
    test(model_path, test_data_path, result_dir, quiet=args.get('--quiet'))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])
