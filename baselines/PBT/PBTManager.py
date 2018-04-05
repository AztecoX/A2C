from baselines.PBT.PBT import PBT

# PBTManager exposes PBT functionality through an interface.
# PBT = Population Based Training.


class PBTManager:
    def __init__(self, flags, config):
        self.pbt = PBT(flags, config)

    def set_up_processes(self):
        self.pbt.set_up_worker_processes()
        self.pbt.start_worker_processes()

    def start_running(self):
        self.pbt.run_workers()

    def handle_requests(self):
        self.pbt.handle_requests()

    def stop_running(self):
        self.pbt.stop_workers()
        self.pbt.finish_worker_processes()

    def wait_for_finish(self):
        self.pbt.finish_worker_processes()

