import os, sys
import tensorflow as tf
from datetime import datetime
from functools import partial

from baselines.a2c.a2c import ActorCriticAgent
from baselines.a2c.runner import Runner
from baselines.common.multienv import SubprocVecEnv, make_sc2env, SingleEnv


class Worker:
    @staticmethod
    def _print(i):
        print(datetime.now())
        print("# batch %d" % i)
        sys.stdout.flush()

    @staticmethod
    def _save_if_training(agent, checkpoint_path, training):
        if training:
            agent.save_default(checkpoint_path)
            agent.flush_summaries()
            sys.stdout.flush()

    def __init__(self, remote, remote_id, flags, config, lock):
        self.remote = remote
        self.id = remote_id
        self.lock = lock
        self.batches_per_pbt_eval = flags.K_batches_per_eval * 1000
        self.envs = []
        self.agent = self.runner = None
        tf.reset_default_graph()
        self.session = tf.Session(config=config.tf_config)
        # Get the Worker unit ready for work.
        Worker.prepare_env_args(flags)
        self.build_envs(Worker.prepare_env_args(flags), flags.n_envs_per_model)
        self.build_agent(flags, config)
        self.build_runner(flags, config)

        # An object for saving and restoring models from storage.
        self.saver = tf.train.Saver()

        # Waiting here for permission to start working.
        if self.can_start_working():
            self.work(flags)
        else:
            remote.close()

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

    def build_envs(self, env_args, envs_per_model):
        self.envs = SubprocVecEnv((partial(make_sc2env, **env_args),) * envs_per_model)
        # return SingleEnv(make_sc2env(**env_args))

    def build_agent(self, flags, config):
        # Set up the agent structure.
        self.agent = ActorCriticAgent(
            session=self.session,
            id=self.id,
            unit_type_emb_dim=5,
            spatial_dim=flags.resolution,
            loss_value_weight=flags.loss_value_weight,
            entropy_weight_action_id=flags.entropy_weight_action,
            entropy_weight_spatial=flags.entropy_weight_spatial,
            scalar_summary_freq=flags.scalar_summary_freq,
            all_summary_freq=flags.all_summary_freq,
            summary_path=(config.full_summary_path + str(self.id)),
            max_gradient_norm=flags.max_gradient_norm,
            optimiser_pars=dict(learning_rate=flags.optimiser_lr,
                                epsilon=flags.optimiser_eps)
        )
        self.agent.build_model()  # Build the agent model
        # TODO this loads last checkpoint model...explore new hyperparameters to
        # TODO differentiate loaded models?
        # TODO also, maybe init is needed to call only once for all agents!
        if os.path.exists(config.full_checkpoint_path):
            self.agent.load_default(config.full_checkpoint_path)
        else:
            self.agent.init()
        return self.agent

    def build_runner(self, flags, config):

        self.runner = Runner(
            envs=self.envs,
            agent=self.agent,
            discount=flags.discount,
            n_steps=flags.n_steps_per_batch,
            checkpoint_path=config.full_checkpoint_path
        )

        self.runner.reset()

        return self.runner

    # Blocking wait for permission to work.
    def can_start_working(self):
        cmd, _ = self.remote.recv()
        return cmd == 'begin'

    def work(self, flags):
        print("Agent n." + str(self.id) + " reporting for duty!", flush=True)

        cmd = ""
        action = ""
        i = 0
        done = False

        if flags.K_batches >= 0:
            n_batches = flags.K_batches * 1000
        else:
            n_batches = -1

        try:
            while True:
                # if remote.poll(): # check for messages from the master process
                # cmd, action = remote.recv()
                if cmd == 'close':
                    break
                else:
                    if i % 1000 == 0:
                        Worker._print(i)
#                    if i % 10000 == 0:
#                        Worker._save_if_training(self.agent,
#                                                 self.runner.checkpoint_path,
#                                                 flags.training)

                    training_input = self.runner.run_batch()  # run

                    if flags.training:
                        self.agent.train(training_input)  # train
                    else:
                        pass

                    i += 1

                    if i % self.batches_per_pbt_eval == 0:
                        done = self.evaluate_and_update_model()

                    if 0 <= n_batches <= i or done:
                        break

        except KeyboardInterrupt:
            pass

        self.remote.send(('done'))
        self.remote.close()

    def evaluate_and_update_model(self):
        # Evaluating...
        self.remote.send(('evaluate',
                          self.id,
                          self.runner.get_and_reset_score()))
        cmd, arg = self.remote.recv()

        # Updating...

        if cmd == 'restore':
            self.agent.load(self.runner.checkpoint_path, arg, self.lock, self.saver)
        elif cmd == 'save':
            self.agent.save(self.runner.checkpoint_path, self.lock, self.saver)
        else:
            return True
        return False
