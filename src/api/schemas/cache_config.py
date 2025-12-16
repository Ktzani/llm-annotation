from pydantic import BaseModel, Field

class CacheConfig(BaseModel):
    enabled: bool = Field(default=True)
    dir: str = Field(default="../../data/.cache")