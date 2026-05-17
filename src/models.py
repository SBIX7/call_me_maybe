"""
Pydantic models for parsing and validating JSON input files.
"""

from pydantic import BaseModel
from typing import Dict, List, Optional


class TypeDef(BaseModel):
    """
    Defines the data type and optional allowed values (enum) for a parameter.
    """
    type: str
    enum: Optional[List[str]] = None


class FunctionDef(BaseModel):
    """
    Represents a callable function's signature, including its
    parameters and return type.
    """
    name: str
    description: str
    parameters: Dict[str, TypeDef]
    returns: TypeDef


class Prompt(BaseModel):
    """
    Represents a single natural language user prompt for function calling.
    """
    prompt: str
