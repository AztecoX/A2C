import os
import shutil
import sys
sys.path.append('../..')
from datetime import datetime
from absl import flags
from baselines.PBT.PBTManager import PBTManager

from config import Config

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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


def main():
    if FLAGS.training:
        check_and_handle_existing_folder(config.full_checkpoint_path)
        check_and_handle_existing_folder(config.full_summary_path)

    if FLAGS.n_models > 1:
        FLAGS.visualize = False


#    _save_if_training(agent)

    pbt = PBTManager(FLAGS, config)
    pbt.set_up_processes()
    pbt.start_running()

#    pbt.wait_for_finish()
    pbt.handle_requests()
    pbt.stop_running()

    print("Okay. Work is done")

#    _print(i)



    #TODO DO NOT CLOSE ENVS BEFORE IT ENDS

#    for env_group in envs:
#        env_group.close()


if __name__ == "__main__":
    main()
