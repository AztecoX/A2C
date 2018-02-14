from baselines.a2c.PBT import PBT

# PBTManager exposes PBT functionality through an interface.
# PBT = Population Based Training.


class PBTManager:
    def __init__(self, flags, config):
        self.pbt = PBT(flags, config)

    def set_up_processes(self):
        self.pbt.set_up_processes()
        self.pbt.initialize_workers()

    def start_running(self):
        self.pbt.run_workers()

    def handle_requests(self):
        self.pbt.handle_requests()

    def stop_running(self):
        self.pbt.stop_workers()
        self.pbt.finish_processes()

    def wait_for_finish(self):
        self.pbt.finish_processes()
        print("Okay.")

