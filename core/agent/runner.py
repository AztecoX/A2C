import numpy as np
import sys
from core.common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from core.common.utils import general_n_step_advantage, combine_first_dimensions
import tensorflow as tf
from absl import flags

class Runner(object):
    def __init__(
            self,
            n_steps=5,
            discount=0.99,
            checkpoint_path=""
    ):
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.n_steps = n_steps
        self.discount = discount
        self.batch_counter = 0
        self.episode_counter = 0
        self.checkpoint_path=checkpoint_path
        self.accumulated_score = 0

    def reset(self, envs):
        obs = envs.reset()
        self.latest_obs = self.obs_processer.process(obs)

    def _log_score_to_tb(self, score, agent):
        summary = tf.Summary()
        summary.value.add(tag='sc2/episode_score', simple_value=score)
        agent.summary_writer.add_summary(summary, self.episode_counter)

    def _handle_episode_end(self, agent, model_id, timestep):
        score = timestep.observation["score_cumulative"][0]
        self._log_score_to_tb(score, agent)
        self.accumulated_score += score
        print("Episode %d ended for model n. %d. Score %f, AccScore %f" % (self.episode_counter, model_id, score, self.accumulated_score))
        self.episode_counter += 1

    def get_and_reset_score(self):
        score = self.accumulated_score
        self.accumulated_score = 0
        return score

    def run_batch(self, envs, agent):
        # init
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((envs.n_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((envs.n_envs, self.n_steps), dtype=np.float32)

        latest_obs = self.latest_obs

        for n in range(self.n_steps):
            action_ids, spatial_action_2ds, value_estimate = agent.step(latest_obs)

            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids, spatial_action_2ds))

            actions_pp = self.action_processer.process(action_ids, spatial_action_2ds)
            obs_raw = envs.step(actions_pp)
            if obs_raw[0] == -1:
                raise KeyboardInterrupt
            latest_obs = self.obs_processer.process(obs_raw)
            mb_rewards[:, n] = [t.reward for t in obs_raw]

            for t in obs_raw:
                if t.last():
                    self._handle_episode_end(agent, agent.id, t)

        mb_values[:, -1] = agent.get_value_estimate(latest_obs)

        n_step_advantage = general_n_step_advantage(
            mb_rewards,
            mb_values,
            self.discount,
            lambda_par=1.0
        )

        full_input = {
            # these are transposed because action/obs
            # processers return [time, env, ...] shaped arrays
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose()
        }
        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()}

        self.latest_obs = latest_obs
        self.batch_counter += 1
        sys.stdout.flush()

        return full_input

