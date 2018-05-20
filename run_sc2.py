import os
import shutil
import sys
sys.path.append('../..')
from datetime import datetime
from absl import flags
from core.PBT.PBTManager import PBTManager

from config import Config

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

config = Config()           # Loading the configuration parameters

FLAGS = flags.FLAGS
FLAGS(sys.argv)             # Parameters now accessible through FLAGS.


def check_and_handle_existing_models(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)
    else:
        if FLAGS.if_output_exists == "continue" or FLAGS.if_output_exists == "continue_individual":
            raise Exception("There are no models to continue from.")

def check_and_handle_existing_summaries(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite" or FLAGS.if_output_exists == "continue_individual":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)

def _print(i):
    print(datetime.now())
    print("# batch %d" % i)
    sys.stdout.flush()


def main():
    if FLAGS.training:
        check_and_handle_existing_models(config.full_checkpoint_path)
        check_and_handle_existing_summaries(FLAGS.summary_path)

    if FLAGS.n_models > 1:
        if FLAGS.if_output_exists == "continue_individual":
            raise Exception("Output set to 'continue_individual'. Either set it to a different setting, or change the number of models to 1.")
        FLAGS.visualize = False

    pbt = PBTManager(FLAGS, config)
    pbt.set_up_processes()
    pbt.start_running()

    pbt.handle_requests()
    pbt.stop_running()

    print("Okay. Work is done")

if __name__ == "__main__":
    main()
