from dataclasses import dataclass

@dataclass
class Config:
    num_clients: int
    num_rounds: int
    epsilon: float
