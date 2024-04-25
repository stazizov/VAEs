from dataclasses import dataclass
import pyrallis


@dataclass
class TrainConfig:
    pass


@pyrallis.wrap()
def train(config: TrainConfig):
    pass
