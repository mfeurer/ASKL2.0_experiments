import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from socketserver import ThreadingMixIn

from multiprocessing import Value

import sys
sys.path.append('..')

from utils import automl_metadata as automl_metadata_task_ids

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


def start_server(call_counter, n_configurations, n_jobs, counters, host, port, run_args):
    handler = lambda *args: PostHandler(call_counter, n_configurations, n_jobs, counters,
                                        run_args, *args)
    server = ThreadedHTTPServer((host, port), handler)
    server.serve_forever()


class Counter:

    def __init__(self):
        self.counter = Value('i', 0)

    def increment(self):
        with self.counter.get_lock():
            self.counter.value += 1

    def get_value(self):
        with self.counter.get_lock():
            return self.counter.value


class PostHandler(BaseHTTPRequestHandler):

    def __init__(self, call_counter, n_configurations, n_jobs, counters, run_args, *args):
        # For reasons I don't understand the assignment has to happen before calling the parent
        # __init__.
        self.call_counter = call_counter
        self.n_configurations = n_configurations
        self.run_args = run_args
        self.n_jobs = n_jobs
        self.counters = counters

        BaseHTTPRequestHandler.__init__(self, *args)

    def do_POST(self):

        # Begin the response
        self.send_response(200)
        self.end_headers()

        # Very basic logging!
        print('Post')

        return

    def do_GET(self):
        path = self.path
        path = path.replace('/?', '', 1)
        path = path.split('&')
        if len(path) != 1:
            return (500, 'Only one argument allow, got %d' % len(path))

        args = {}
        for pair in path:
            arg, value = pair.split('=')
            args[arg] = value

        task_id = int(args['task_id'])
        # Where does this come from??
        key = task_id

        try:
            value = self.counters[key].get_value()
        except KeyError:
            print('Legal keys', self.counters.keys())
            raise

        # Begin the response
        self.send_response(200)
        self.end_headers()

        if value >= self.n_configurations:
            value = -1
        else:
            self.counters[key].increment()
            self.call_counter.increment()

        call_count = self.call_counter.get_value()

        response = {
            'task_id': task_id,
            'run_args': self.run_args,
            'counter': value,
            'call_count': call_count,
            'n_jobs': self.n_jobs,
        }
        response_string = json.dumps(response, indent=4)
        self.wfile.write(response_string.encode('utf8'))

        # Very basic logging!
        print('Get', key, call_count, value)

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to folder with incumbents.json, space.json and task_to_id.json.')
    parser.add_argument('--searchspace', choices=("full", "iterative"), required=True)
    parser.add_argument('--evaluation', choices=("holdout", "CV"), required=True)
    parser.add_argument('--cv', choices=(3, 5, 10), type=int, required=False)  # depends
    parser.add_argument("--iterative-fit", choices=("True", "False"), required=True)
    parser.add_argument("--early-stopping", choices=("True", "False"), required=True)
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    args = parser.parse_args()

    configurations_file = os.path.join(args.input_dir, "incumbents.json")
    configurationspace_file = os.path.join(args.input_dir, "space.json")

    evaluation = args.evaluation
    iterative_fit = args.iterative_fit == "True"
    early_stopping = args.early_stopping == "True"
    if ("_nif" in args.input_dir and iterative_fit)\
            or ("_if" in args.input_dir and not iterative_fit):
        raise ValueError("Wrong early stopping!", args.input_dir, iterative_fit)
    if ("_nes" in args.input_dir and early_stopping)\
            or ("_es" in args.input_dir and not early_stopping):
        raise ValueError("Wrong iterative_fit!: ", args.input_dir, early_stopping)
    if args.searchspace not in args.input_dir:
        raise ValueError("Wrong searchspace!: ", args.input_dir, args.searchspace)

    host = args.host
    port = args.port

    with open(configurations_file) as fh:
       configurations = json.load(fh)

    call_counter = Counter()
    n_jobs = len(configurations) * len(automl_metadata_task_ids)
    counters = {task_id: Counter() for task_id in automl_metadata_task_ids}

    print('Starting server for:')
    print('    %d configurations' % len(configurations))
    print('    %d tasks' % len(automl_metadata_task_ids))
    print('    %d total' % n_jobs)

    start_server(
        call_counter=call_counter,
        n_configurations=len(configurations),
        n_jobs=n_jobs,
        counters=counters,
        run_args=(evaluation, iterative_fit, early_stopping, args.cv, args.searchspace),
        host=host,
        port=port,
    )
