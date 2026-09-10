"""What a worker process runs.

Deliberately free of imports: a spawned child imports this module to unpickle
its target, and importing the sweep would pull in every simulator with it.
"""


def send_outcome(worker, experiment_config: dict, sender) -> None:
    sender.send(worker(experiment_config))
    sender.close()
