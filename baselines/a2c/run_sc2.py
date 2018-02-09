import logging
import sys
import os
import shutil
import sys
from datetime import datetime
from functools import partial
import tensorflow as tf
from absl import flags
from baselines.a2c.a2c import ActorCriticAgent
from baselines.a2c.runner import Runner
from baselines.common.multienv import SubprocVecEnv, make_sc2env, SingleEnv
from baselines.a2c.config import Config
import time

config = Config()           # Loading the configuration parameters

FLAGS = flags.FLAGS
FLAGS(sys.argv)             # Parameters now accessible through FLAGS.


def check_and_handle_existing_folder(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)


def _print(i):
    print(datetime.now())
    print("# batch %d" % i)
    sys.stdout.flush()


def _save_if_training(agent):
    if FLAGS.training:
        agent.save(config.full_checkpoint_path)
        agent.flush_summaries()
        sys.stdout.flush()

def main_loop(model_id, agent, runner, n_batches):
    i = 0

    # Main loop for running and training
    try:
        while True:
            if i % 500 == 0:
                _print(i)
            if i % 4000 == 0:
                _save_if_training(agent)

            training_input = runner.run_batch()     # run

            if FLAGS.training:
                agent.train(training_input)         # train
            else:
                pass

            i += 1
            if 0 <= n_batches <= i:
                break
    except KeyboardInterrupt:
        pass

    print("Okay. Work is done")

    _print(i)


def main():
    if FLAGS.training:
        check_and_handle_existing_folder(config.full_checkpoint_path)
        check_and_handle_existing_folder(config.full_summary_path)

    # Set up the environments (each separate environment is a subprocess)
    env_args = dict(
        map_name=FLAGS.map_name,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        screen_size_px=(FLAGS.resolution,) * 2,
        minimap_size_px=(FLAGS.resolution,) * 2,
        visualize=FLAGS.visualize
    )

    tf.reset_default_graph()
    session = tf.Session()

    agents = []
    runners = []
    envs = []
    for n in range(FLAGS.n_models):
        # Set up the agent structure.

        env_group = SubprocVecEnv((partial(make_sc2env, **env_args),) * FLAGS.n_envs_per_model)
        # env_group = SingleEnv(make_sc2env(**env_args))

        envs.append(env_group)

        agent = ActorCriticAgent(
            session=session,
            id=n,
            unit_type_emb_dim=5,
            spatial_dim=FLAGS.resolution,
            loss_value_weight=FLAGS.loss_value_weight,
            entropy_weight_action_id=FLAGS.entropy_weight_action,
            entropy_weight_spatial=FLAGS.entropy_weight_spatial,
            scalar_summary_freq=FLAGS.scalar_summary_freq,
            all_summary_freq=FLAGS.all_summary_freq,
            summary_path=(config.full_summary_path + str(1)),
            max_gradient_norm=FLAGS.max_gradient_norm,
            optimiser_pars=dict(learning_rate=FLAGS.optimiser_lr,
                                epsilon=FLAGS.optimiser_eps)
        )
        agent.build_model() # Build the agent model

        # TODO this loads last checkpoint...explore different models?
        # TODO also, maybe init is needed to call only once for all agents!
        if os.path.exists(config.full_checkpoint_path):
            agent.load(config.full_checkpoint_path)
        else:
            agent.init()

        runner = Runner(
            envs=env_group,
            agent=agent,
            discount=FLAGS.discount,
            n_steps=FLAGS.n_steps_per_batch,
        )

        runner.reset()

        agents.append(agent)
        runners.append(runner)

        print("Created " + str(n) + " out of " + str(FLAGS.n_models) + " agents.")

    if FLAGS.K_batches >= 0:
        n_batches = FLAGS.K_batches * 1000
    else:
        n_batches = -1

    print("Created " + str(len(agents) + 1) + " agents and " + (str(len(envs)) * FLAGS.n_envs_per_model) + "envs.")
#    _save_if_training(agent)

    i = 0

    # Main loop for running and training
    try:
        while True:
            if i % 500 == 0:
                _print(i)
            if i % 4000 == 0:
                _save_if_training(agent)

            training_input = runners[0].run_batch()     # run

            if FLAGS.training:
                agents[0].train(training_input)         # train
            else:
                pass

            i += 1
            if 0 <= n_batches <= i:
                break
    except KeyboardInterrupt:
        pass

    print("Okay. Work is done")

    _print(i)



    #TODO DO NOT CLOSE ENVS BEFORE IT ENDS

    for env_group in envs:
        env_group.close()


if __name__ == "__main__":
    main()