from pydantic import BaseModel
from typing import Dict, List, Optional


class TypeDef(BaseModel):
    type: str
    enum: Optional[List[str]] = None


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, TypeDef]
    returns: TypeDef


class Prompt(BaseModel):
    prompt: str
