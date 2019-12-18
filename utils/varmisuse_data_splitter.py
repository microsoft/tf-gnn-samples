#!/usr/bin/env python
"""
Usage:
   varmisuse_data_splitter.py [options] RAW_DATA_DIR OUT_DIR

Reads in datapoints from a set of files and creates smaller files mixing these, in a format
suitable for streaming them into the training process.

Options:
    -h --help                       Show this screen.
    --chunk-size NUM                Number of samples per output file. [default: 100]
    --num-workers NUM               Number of worker processes. Defaults to number of CPU cores.
    --window-size NUM               Number of samples to load before mixing and writing things out. [default: 5000]
    --azure-info=<path>             Azure authentication information file (JSON). [default: azure_auth.json]
    --debug                         Turn on debugger.
"""
from typing import List, Any

import numpy as np
from more_itertools import chunked
from docopt import docopt
from dpu_utils.utils import run_and_debug, RichPath
from multiprocessing import Process, Queue, cpu_count


def _data_loading_worker(file_queue: Queue, result_queue: Queue) -> None:
    while True:
        next_path = file_queue.get()
        if next_path is None:  # Our signal that all files have been processed
            file_queue.put(None)  # Signal to the other workers
            result_queue.put(None)  # Signal to the controller that we are done
            break

        # Read the file and push examples out as soon as we get them:
        for raw_sample in next_path.read_by_file_suffix():
            result_queue.put(raw_sample)


def _write_data(out_dir: RichPath, window_idx: int, chunk_size: int, data_window: List[Any]):
    np.random.shuffle(data_window)
    for chunk_idx, data_chunk in enumerate(chunked(data_window, chunk_size)):
        out_file = out_dir.join('chunk_%i-%i.jsonl.gz' % (window_idx, chunk_idx))
        out_file.save_as_compressed_file(data_chunk)


def run(args):
    azure_info_path = args.get('--azure-info', None)
    in_dir = RichPath.create(args['RAW_DATA_DIR'], azure_info_path)
    out_dir = RichPath.create(args['OUT_DIR'], azure_info_path)
    out_dir.make_as_dir()

    num_workers = int(args.get('--num-workers') or cpu_count())
    chunk_size = int(args['--chunk-size'])
    window_size = int(args['--window-size'])

    files_to_load = list(in_dir.iterate_filtered_files_in_dir("*.gz"))
    path_queue = Queue(maxsize=len(files_to_load) + 1)
    result_queue = Queue(1000)

    # Set up list of work to do:
    for path in files_to_load:
        path_queue.put(path)
    path_queue.put(None)  # Signal for the end of the queue

    # Set up workers:
    workers = []
    for _ in range(num_workers):
        workers.append(Process(target=_data_loading_worker,
                               args=(path_queue, result_queue,)))
        workers[-1].start()

    # Consume the data:
    num_workers_terminated = 0
    data_window = []
    window_idx = 0
    while num_workers_terminated < len(workers):
        parsed_sample = result_queue.get()
        if parsed_sample is None:
            num_workers_terminated += 1  # Worker signaled that it's done
        else:
            data_window.append(parsed_sample)
            if len(data_window) >= window_size:
                _write_data(out_dir, window_idx, chunk_size, data_window)
                data_window = []
                window_idx += 1

    # Write out the remainder of the data:
    _write_data(out_dir, window_idx, chunk_size, data_window)

    # Clean up the workers:
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), enable_debugging=args['--debug'])
