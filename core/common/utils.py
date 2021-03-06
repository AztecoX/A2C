import numpy as np
import tensorflow as tf


def weighted_random_sample(weights):
    u = tf.random_uniform(tf.shape(weights))
    return tf.argmax(tf.log(u) / weights, axis=1)


def select_from_each_row(params, indices):
    sel = tf.stack([tf.range(tf.shape(params)[0]), indices], axis=1)
    return tf.gather_nd(params, sel)


def calculate_n_step_reward(
        one_step_rewards: np.ndarray,
        discount: float,
        last_state_values: np.ndarray):

    discount = discount ** np.arange(one_step_rewards.shape[1], -1, -1)
    reverse_rewards = np.c_[one_step_rewards, last_state_values][:, ::-1]
    full_discounted_reverse_rewards = reverse_rewards * discount
    return (np.cumsum(full_discounted_reverse_rewards, axis=1) / discount)[:, :0:-1]


def general_n_step_advantage(
        one_step_rewards: np.ndarray,
        value_estimates: np.ndarray,
        discount: float,
        lambda_par: float
):
    assert 0.0 < discount <= 1.0
    assert 0.0 <= lambda_par <= 1.0
    batch_size, timesteps = one_step_rewards.shape
    assert value_estimates.shape == (batch_size, timesteps + 1)
    delta = one_step_rewards + discount * value_estimates[:, 1:] - value_estimates[:, :-1]

    if lambda_par == 0:
        return delta

    delta_rev = delta[:, ::-1]
    adjustment = (discount * lambda_par) ** np.arange(timesteps, 0, -1)
    advantage = (np.cumsum(delta_rev * adjustment, axis=1) / adjustment)[:, ::-1]
    return advantage


def combine_first_dimensions(x: np.ndarray):
    first_dim = x.shape[0] * x.shape[1]
    other_dims = x.shape[2:]
    dims = (first_dim,) + other_dims
    return x.reshape(*dims)


def ravel_index_pairs(idx_pairs, n_col):
    return tf.reduce_sum(idx_pairs * np.array([n_col, 1])[np.newaxis, ...], axis=1)


def dict_of_lists_to_list_of_dicst(x: dict):
    dim = {len(v) for v in x.values()}
    assert len(dim) == 1
    dim = dim.pop()
    return [{k: x[k][i] for k in x} for i in range(dim)]