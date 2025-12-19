from typing import Optional
from src.api.schemas.enums import PromptType

from src.config.prompts import (
    BASE_ANNOTATION_PROMPT,
    FEW_SHOT_PROMPT,
    COT_PROMPT,
    SIMPLER_PROMPT,
)


def get_prompt_template(
    prompt_type: PromptType,
    custom_prompt: Optional[str] = None,
) -> str:
    if custom_prompt:
        return custom_prompt

    prompt_map = {
        PromptType.BASE: BASE_ANNOTATION_PROMPT,
        PromptType.FEW_SHOT: FEW_SHOT_PROMPT,
        PromptType.COT: COT_PROMPT,
        PromptType.SIMPLER: SIMPLER_PROMPT,
    }

    return prompt_map[prompt_type]
