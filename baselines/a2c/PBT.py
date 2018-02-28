from multiprocessing import Process, Pipe, Lock, connection
import numpy as np
from baselines.a2c.Worker import Worker
from baselines.common.multienv import SubprocVecEnv, make_sc2env, SingleEnv
from functools import partial


class PBT:
    def __init__(self, flags, config):
        self.ps = self.remotes = self.work_remotes = []
        self.flags = flags
        self.config = config
        self.lock = Lock() # lock for .ckpt models file access
        self.env_groups = self.build_envs(Worker.prepare_env_args(self.flags))

    @staticmethod
    def prepare_env_args(flags):
        return dict(
            map_name=flags.map_name,
            step_mul=flags.step_mul,
            game_steps_per_episode=0,
            screen_size_px=(flags.resolution,) * 2,
            minimap_size_px=(flags.resolution,) * 2,
            visualize=flags.visualize
        )

    def build_envs(self, env_args):
        env_groups = []
        for i in range(self.flags.n_models):
            env_groups.append(SubprocVecEnv((partial(make_sc2env, **env_args),)
                                  * self.flags.n_envs_per_model))
        return env_groups

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
                       self.env_groups[id_assigned],
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
                           self.lock,
                           self.env_groups[n])

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
            # Master checks for Worker requests and resolves them.
            requests = connection.wait(self.remotes, timeout=5)
            if len(requests) != 0:
                for r in requests:
                    msg, worker_id, arg = r.recv()
                    if msg == 'evaluate':
                        scores[worker_id] = arg
                        is_underperforming, outperforming_model = self.is_model_underperforming(scores, worker_id)
                        if is_underperforming:
                            # Current model not good enough, exploit+explore a better one.
                            r.send(('restore', outperforming_model))
                        else:
                            # Current model worth training further.
                            r.send(('save', None))
                    elif msg == 'yield':
                        # The process has to be restarted to make space for
                        # a new one, since tf does only
                        # release GPU resources correctly on process exit.
                        self.restart_process(worker_id, np.argmax(scores), arg)
                    elif msg == 'done':
                        self.finish_processes()

    def stop_workers(self):
        for r in self.remotes:
            r.send(('close', None))

    def start_process(self, outperformed_id, episode_counter=0):
        self.ps[outperformed_id].start()
        self.remotes[outperformed_id].send(('begin', episode_counter))

    def restart_process(self, outperformed_id, outperforming_id, episode_counter):
        self.ps[outperformed_id].join()

        self.reset_process(outperformed_id, outperforming_id)

        self.start_process(outperformed_id, episode_counter)
        # TODO build the new one based on outperforming one.

    def finish_processes(self):
        for p in self.ps:
            p.join()

    def is_model_underperforming(self, scores, worker_id):
        if self.flags.exploration_threshold_metric == "20_percent_top_and_bottom":
            return self.exploration_20_percent_metric(scores, worker_id)
        else:
            return False, -1

    def exploration_20_percent_metric(self, scores, worker_id):
        threshold_modifier = 5.0                # 20% means roughly every fifth model
        if len(scores) <= threshold_modifier:   # If there are 5 or less models, it is straightforward
            if worker_id == np.argmin(scores):
                return True, np.argmax(scores)
            else:
                return False, -1
        else:
            counter = 0
            comparison_counter = 0
            # Threshold decides how many models fit into the top and bottom 20% category.
            threshold = int(np.ceil(len(scores) / threshold_modifier))
            while counter < len(scores):
                if (scores[counter] < scores[worker_id]) and (worker_id != counter):
                    comparison_counter += 1
                counter += 1

            if comparison_counter < threshold:
                # It has been determined, that the model will be replaced.
                # Now it has to be decided, which one will replace it.

                # Getting the best performing candidates...
                candidates = self.get_best_performing_candidates(scores, threshold)

                # Choosing one of them at random:
                return True, candidates[np.random.randint(0, threshold - 1)]
            else:
                return False, -1

    @staticmethod
    def get_best_performing_candidates(scores, threshold):
        candidates = []
        sc = list(scores)
        for i in range(threshold - 1):
            x = np.argmax(sc)
            candidates.append(x)
            sc[x] = -1

        return candidates
