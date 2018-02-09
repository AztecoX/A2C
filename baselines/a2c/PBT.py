from multiprocessing import Process, Pipe
from baselines.a2c.Worker import Worker


class PBT:
    def __init__(self, flags, config):
        self.ps = self.remotes = self.work_remotes = []
        self.flags = flags
        self.config = config

    def set_up_processes(self):
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.flags.n_models)])

        for n in range(self.flags.n_models):
            worker_args = (self.work_remotes[n],
                           n,
                           self.flags,
                           self.config)

            self.ps.append(Process(target=Worker,
                                   args=worker_args))
        return self.remotes, self.ps

    def initialize_workers(self):
        for p in self.ps:
            p.start()

    def run_workers(self):
        for r in self.remotes:
            r.send(('begin', None))

    def stop_workers(self):
        for r in self.remotes:
            r.send(('close', None))

    def finish_processes(self):
        for p in self.ps:
            p.join()
