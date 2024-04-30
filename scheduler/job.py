from dataclasses import dataclass, field, astuple
from itertools import count
import numpy as np


@dataclass
class Job:
    p: float  # Processing time
    d: np.array
    i: int = None  # Assigned machine
    S: float = None  # Starting time
    w: float = 1  # Weight
    r: float = 0  # Release time

    id: int = field(default_factory=count().__next__) # Starts at 0

    # Equality check required for intervaltree
    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id
