from pydantic import BaseModel
from typing import Dict


class TypeDef(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, TypeDef]
    returns: TypeDef


class Prompt(BaseModel):
    prompt: str
