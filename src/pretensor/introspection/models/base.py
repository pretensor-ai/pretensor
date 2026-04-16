from pydantic import BaseModel, ConfigDict


class PretensorModel(BaseModel):
    model_config = ConfigDict(frozen=True)
