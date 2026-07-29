import re
from dataclasses import dataclass

@dataclass
class Candle:
    open: int
    high: int
    low: int
    close: int