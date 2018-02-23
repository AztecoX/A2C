from multiprocessing import Process, Pipe, Lock, connection
import numpy as np
from baselines.a2c.Worker import Worker


class PBT:
    def __init__(self, flags, config):
        self.ps = self.remotes = self.work_remotes = []
        self.flags = flags
        self.config = config
        self.lock = Lock() # lock for .ckpt models file access

    def reset_process(self, id_assigned, id_outperforming):
        remote, work_remote = Pipe()

        ps = list(self.ps)
        remotes = list(self.remotes)
        work_remotes = list(self.work_remotes)

        del ps[id_assigned]
        del remotes[id_assigned]
        del work_remotes[id_assigned]

        remotes.insert(id_assigned, remote)
        work_remotes.insert(id_assigned, work_remote)

        self.remotes = tuple(remotes)
        self.work_remotes = tuple(work_remotes)

        worker_args = (self.work_remotes[id_assigned],
                       id_assigned,
                       self.flags,
                       self.config,
                       self.lock,
                       True,
                       id_outperforming
                       )

        ps.insert(id_assigned, Process(target=Worker,
                           args=worker_args))

        self.ps = tuple(ps)

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
                    elif msg == 'yield':
                        # The process has to be restarted to make space for
                        # a new one, since tf does only
                        # release GPU resources correctly on process exit.
                        self.restart_process(r, worker_id, np.argmax(scores))
                    elif msg == 'done':
                        self.finish_processes()

    def stop_workers(self):
        for r in self.remotes:
            r.send(('close', None))

    def start_process(self, outperformed_id):
        self.ps[outperformed_id].start()
        self.remotes[outperformed_id].send(('begin', None))

    def restart_process(self, remote, outperformed_id, outperforming_id):
        self.ps[outperformed_id].join()

        self.reset_process(outperformed_id, outperforming_id)

        self.start_process(outperformed_id)
        # TODO build the new one based on outperforming one.

    def finish_processes(self):
        for p in self.ps:
            p.join()
