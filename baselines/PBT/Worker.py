import os, sys, time
import numpy as np
import tensorflow as tf
from datetime import datetime
from functools import partial

from baselines.agent.agent import ActorCriticAgent
from baselines.agent.runner import Runner


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

    def __init__(self, remote, remote_id, flags, config, lock, envs, rebuilding=False, outperforming_id=-1, step_counter=0):
        self.remote = remote
        self.id = remote_id
        self.lock = lock
        self.config = config
        self.flags = flags
        self.batches_per_eval = flags.K_batches_per_eval * 1000
        self.envs = envs
        self.agent = self.runner = None
        self.step_counter = 0
        tf.reset_default_graph()

        self.global_step_tensor = tf.Variable(step_counter, trainable=False, name="global_step")
        self.session = tf.Session(config=self.config.tf_config)

        # An object for saving and restoring models from storage.
        self.saver = None

        if self.flags.randomize_hyperparams:
            self.randomize_hyperparams()
        else:
            self.initialize_hyperparams()

        self.build_agent(rebuilding, outperforming_id, step_counter)
        self.build_runner(self.flags, self.config)

        # Waiting here for permission to start working.
        if self.can_start_working():
            self.work(self.flags)
        else:
            remote.close()

    def build_agent(self, rebuilding=False, outperforming_model_id=0, step_counter=0):
        # Set up the agent structure.
        self.agent = ActorCriticAgent(
            session=self.session,
            id=self.id,
            unit_type_emb_dim=5,
            spatial_dim=self.flags.resolution,
            loss_value_weight=self.loss_value_weight,
            entropy_weight_action_id=self.entropy_weight_action,
            entropy_weight_spatial=self.entropy_weight_spatial,
            scalar_summary_freq=self.flags.scalar_summary_freq,
            all_summary_freq=self.flags.all_summary_freq,
            summary_path=(self.config.full_summary_path + str(self.id)),
            max_gradient_norm=self.max_gradient_norm,
            optimiser_pars=dict(learning_rate=self.optimiser_lr,
                                epsilon=self.optimiser_eps)
        )

        prev_global_step = tf.Variable(step_counter, name='prev_global_step', trainable=False, dtype=tf.int32)

        self.agent.build_model()  # Build the agent model

        # An object for saving and restoring models from storage.
        self.saver = tf.train.Saver()

        if rebuilding:
            self.agent.load(self.config.full_checkpoint_path, outperforming_model_id, self.lock, self.saver)

            initial_global_step = tf.assign(tf.train.get_global_step(), prev_global_step)
            print("initial_global_step: %s" % initial_global_step)
            self.session.run(initial_global_step)
            print("GLOBAL STEP AFTER: %s" % self.session.run(tf.train.get_global_step()))
#            tf.train.global_step.assign(self.sess, self.global_step_tensor))
            self.agent.update_train_step(step_counter)
            var = [v for v in tf.trainable_variables() if v.name == "theta/spatial_action/weights:0"][0][0][0][0][0]
            print("REBUILT MODEL " + str(self.id) + ", BASED ON MODEL " + str(outperforming_model_id) + ", WEIGHT VALUE: " + str(self.session.run(var)))
        elif os.path.exists(self.config.full_checkpoint_path):
            self.agent.load_default(self.config.full_checkpoint_path)
        else:
            self.agent.init()

        return self.agent

    def randomize_hyperparams(self):
        self.max_gradient_norm = np.random.uniform(self.flags.min_max_gradient_norm, self.flags.max_max_gradient_norm)
        self.discount = np.random.uniform(self.flags.min_discount, self.flags.max_discount)
        self.loss_value_weight = np.random.uniform(self.flags.min_loss_value_weight, self.flags.max_loss_value_weight)
        self.optimiser_lr = np.random.uniform(self.flags.min_optimiser_lr, self.flags.max_optimiser_lr)
        self.optimiser_eps = np.random.uniform(self.flags.min_optimiser_eps, self.flags.max_optimiser_eps)
        self.entropy_weight_spatial = np.random.uniform(self.flags.min_entropy_weight_spatial, self.flags.max_entropy_weight_spatial)
        self.entropy_weight_action = np.random.uniform(self.flags.min_entropy_weight_action, self.flags.max_entropy_weight_action)

    def initialize_hyperparams(self):
        self.max_gradient_norm = self.flags.max_gradient_norm
        self.discount = self.flags.discount
        self.loss_value_weight = self.flags.loss_value_weight
        self.optimiser_lr = self.flags.optimiser_lr
        self.optimiser_eps = self.flags.optimiser_eps
        self.entropy_weight_spatial = self.flags.entropy_weight_spatial
        self.entropy_weight_action = self.flags.entropy_weight_action

    def build_runner(self, flags, config):

        self.runner = Runner(
            discount=self.discount,
            n_steps=flags.n_steps_per_batch,
            checkpoint_path=config.full_checkpoint_path
        )

        self.runner.reset(self.envs)

        return self.runner

    # Blocking wait for permission to work.
    def can_start_working(self):
        cmd, episode_counter, step_counter = self.remote.recv()
        self.runner.episode_counter = episode_counter
        self.step_counter=step_counter
        return cmd == 'begin'

    def work(self, flags):
        print("Agent n." + str(self.id) + " reporting for duty!", flush=True)
        self.agent.save(self.runner.checkpoint_path, self.lock, self.saver)
        cmd = ""
        action = ""
        i = self.step_counter
        done = False

        try:
            while True:
                if cmd == 'close':
                    break
                else:
                    if i % 1000 == 0:
                        Worker._print(i)

                    training_input = self.runner.run_batch(self.envs, self.agent)  # run

                    if flags.training:
                        self.agent.train(training_input)  # train
                    else:
                        pass

                    i += 1

                    if i % self.batches_per_eval == 0:
                        done = self.evaluate_and_update_model(i)

                    if 0 <= flags.episodes <= self.runner.episode_counter or done:
                        break

        except KeyboardInterrupt:
            pass

        if not done:
            self.remote.send(('done', self.id, None, None))
        self.remote.close()

    def load_better_model(self, step_counter):
        self.remote.send(('yield', self.id, self.runner.episode_counter, step_counter))

    def evaluate_and_update_model(self, step_counter):
        # Evaluating...
        self.remote.send(('evaluate',
                          self.id,
                          self.runner.get_and_reset_score(), None))
        cmd, arg, _ = self.remote.recv()

        # Updating...

        if cmd == 'restore':
            self.load_better_model(step_counter)
            return True
        elif cmd == 'save':
            self.agent.save(self.runner.checkpoint_path, self.lock, self.saver)
            return False
        else:
            return True
