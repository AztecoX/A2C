import sys, os
import tensorflow as tf
from absl import flags

class Config:
    def __init__(self):
        self.full_checkpoint_path = ""
        self.full_summary_path = ""
        self.set_flags()   # Set up the configuration.
        self.set_paths()
        self.set_gpu_usage()

    # The main config setup is here.
    def set_flags(self):
        flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
        flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
        flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
        flags.DEFINE_integer("n_models", 1, "Number of models being trained in parallel by PBT.")
        flags.DEFINE_integer("n_envs_per_model", 4, "Number of environments to run in parallel per model.")
        flags.DEFINE_integer("n_steps_per_batch", 8,
                             "Number of steps per batch.")
        flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
        flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
        flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
        flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
        flags.DEFINE_string("model_name", "temp_testing", "Name for checkpoints and tensorboard summaries")
        flags.DEFINE_integer("episodes", 25000,
                             "Number of training batches to run in thousands, use -1 to run forever")
        flags.DEFINE_integer("K_batches_per_eval", 10,
                             "Number of training batches to run in thousands before PBT evaluates model")
        flags.DEFINE_string("map_name", "CollectMineralShards", "Name of a map to use.")
        flags.DEFINE_boolean("training", True,
                             "if should train the model, if false then save only episode score summaries"
                             )
        flags.DEFINE_enum("if_output_exists", "overwrite", ["fail", "overwrite", "continue"],
                          "What to do if summary and model output exists, only for training, is ignored if notraining")

        # The following parameters are considering a potential randomization.
        flags.DEFINE_boolean("randomize_hyperparams", True, "Without randomization, using PBT does not make much sense.")

        flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")
        flags.DEFINE_float("min_max_gradient_norm", 150.0, "good value might depend on the environment")
        flags.DEFINE_float("max_max_gradient_norm", 700.0, "good value might depend on the environment")

        flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
        flags.DEFINE_float("min_discount", 0.90, "Reward-discount for the agent")
        flags.DEFINE_float("max_discount", 0.999, "Reward-discount for the agent")

        flags.DEFINE_float("loss_value_weight", 1.0, "good value might depend on the environment")
        flags.DEFINE_float("min_loss_value_weight", 0.8, "good value might depend on the environment")
        flags.DEFINE_float("max_loss_value_weight", 1.2, "good value might depend on the environment")

        flags.DEFINE_float("optimiser_lr", 1e-4, "Optimiser learning rate")
        flags.DEFINE_float("min_optimiser_lr", 1e-7, "Optimiser learning rate")
        flags.DEFINE_float("max_optimiser_lr", 1e-3, "Optimiser learning rate")

        flags.DEFINE_float("optimiser_eps", 1e-7, "Optimiser parameter preventing by-zero division.")
        flags.DEFINE_float("min_optimiser_eps", 1e-9, "Optimiser parameter preventing by-zero division.")
        flags.DEFINE_float("max_optimiser_eps", 1e-5, "Optimiser parameter preventing by-zero division.")

        flags.DEFINE_float("entropy_weight_spatial", 1e-4, "entropy of spatial action distribution loss weight")
        flags.DEFINE_float("min_entropy_weight_spatial", 1e-6, "entropy of spatial action distribution loss weight")
        flags.DEFINE_float("max_entropy_weight_spatial", 1e-2, "entropy of spatial action distribution loss weight")

        flags.DEFINE_float("entropy_weight_action", 1e-4, "entropy of action-id distribution loss weight")
        flags.DEFINE_float("min_entropy_weight_action", 1e-6, "entropy of action-id distribution loss weight")
        flags.DEFINE_float("max_entropy_weight_action", 1e-2, "entropy of action-id distribution loss weight")

# Exploration related variables
        flags.DEFINE_enum("exploitation_threshold_metric", "20_percent_top_and_bottom", ["20_percent_top_and_bottom", "Nothing"],
                          "What is the threshold for underperforming model recognition.")
        flags.DEFINE_float("exploitation_worth_percentage", 0.10,
                          "The best model has to outperform the current one by 'exploitation_worth_percentage'"
                          " percent to consider replacing it.")


    def set_paths(self):
        FLAGS = flags.FLAGS
        FLAGS(sys.argv)

        self.full_checkpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name)

        if FLAGS.training:
            self.full_summary_path = os.path.join(FLAGS.summary_path, FLAGS.model_name)
        else:
            self.full_summary_path = os.path.join(FLAGS.summary_path, "no_training", FLAGS.model_name)

    def set_gpu_usage(self):
        # Dividing memory fairly among the processes.
        FLAGS = flags.FLAGS
        self.tf_config = tf.ConfigProto()
        # Feel free to experiment with the GPU memory allocation. Some safe-lock has to be
        # there though due to possibility of one model allocating more GPU memory than the
        # other. In that case it could be possible, that when replacing a less memory
        # demanding model with a more demanding one, we could run out of memory. I did not
        # make a better workaround yet.
#        self.tf_config.gpu_options.allow_growth = True
        self.tf_config.gpu_options.per_process_gpu_memory_fraction = 0.60 / FLAGS.n_models
