from pydantic import BaseModel
from typing import List, Optional


class FictionModel(BaseModel):
    item_id: str
    genre: str
