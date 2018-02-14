from multiprocessing import Process, Pipe, Lock, connection
import numpy as np
from baselines.a2c.Worker import Worker


class PBT:
    def __init__(self, flags, config):
        self.ps = self.remotes = self.work_remotes = []
        self.flags = flags
        self.config = config
        self.lock = Lock() # lock for .ckpt models file access

    def set_up_processes(self):
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.flags.n_models)])

        for n in range(self.flags.n_models):
            worker_args = (self.work_remotes[n],
                           n,
                           self.flags,
                           self.config,
                           self.lock)

            self.ps.append(Process(target=Worker,
                                   args=worker_args))
        return self.remotes, self.ps

    def initialize_workers(self):
        for p in self.ps:
            p.start()

    def run_workers(self):
        for r in self.remotes:
            r.send(('begin', None))

    def handle_requests(self):
        scores = np.zeros(self.flags.n_models, dtype=int)
        while True:
            # Master checks for Worker requests.
            requests = connection.wait(self.remotes, timeout=5)
            if len(requests) != 0:
                for r in requests:
                    msg, worker_id, score = r.recv()
                    if msg == 'evaluate':
                        scores[worker_id] = score
                        if np.median(scores) > score:
                            # Current model not good enough,
                            # exploit+explore a better one.
                            r.send(('restore', np.argmax(scores)))
                        else:
                            # Current model worth training further.
                            r.send(('save', None))
                    elif msg == 'done':
                        self.finish_processes()




    def stop_workers(self):
        for r in self.remotes:
            r.send(('close', None))

    def finish_processes(self):
        for p in self.ps:
            p.join()
