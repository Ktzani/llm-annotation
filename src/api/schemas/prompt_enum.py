from enum import Enum

class PromptType(str, Enum):
    BASE = "base"
    FEW_SHOT = "few_shot"
    COT = "chain_of_thought"
    SIMPLER = "simpler"
