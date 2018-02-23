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

    def set_flags(self):
        flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
        flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
        flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
        flags.DEFINE_integer("n_models", 2, "Number of models being trained in parallel by PBT.")
        flags.DEFINE_integer("n_envs_per_model", 1, "Number of environments to run in parallel")
        flags.DEFINE_integer("n_steps_per_batch", 8,
                             "Number of steps per batch, EXPERIMENT WITH THIS?")
        flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
        flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
        flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
        flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
        flags.DEFINE_string("model_name", "temp_testing", "Name for checkpoints and tensorboard summaries")
        flags.DEFINE_integer("K_batches", 10,
                             "Number of training batches to run in thousands, use -1 to run forever")
        flags.DEFINE_integer("K_batches_per_eval", 1,
                             "Number of training batches to run in thousands before PBT evaluates model")
        flags.DEFINE_string("map_name", "MoveToBeacon", "Name of a map to use.")
        flags.DEFINE_boolean("training", True,
                             "if should train the model, if false then save only episode score summaries"
                             )
        flags.DEFINE_enum("if_output_exists", "overwrite", ["fail", "overwrite", "continue"],
                          "What to do if summary and model output exists, only for training, is ignored if notraining")

        flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")

        flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
        flags.DEFINE_float("loss_value_weight", 1.0, "good value might depend on the environment")
        flags.DEFINE_float("optimiser_lr", 1e-4, "Optimiser learning rate")
        flags.DEFINE_float("optimiser_eps", 1e-7, "Optimiser parameter preventing by-zero division.")
        flags.DEFINE_float("entropy_weight_spatial", 1e-4,
                          "entropy of spatial action distribution loss weight")
        flags.DEFINE_float("entropy_weight_action", 1e-4, "entropy of action-id distribution loss weight")


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
        self.tf_config.gpu_options.allow_growth = True # Necessary, otherwise the processes eat up everything.
        self.tf_config.gpu_options.per_process_gpu_memory_fraction = 1.0 / FLAGS.n_models