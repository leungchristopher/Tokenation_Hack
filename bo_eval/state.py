"""Inspect glue: the per-sample Session lives in the sample store."""

from typing import Callable, TypeVar

from inspect_ai.tool import ToolError
from inspect_ai.util import StoreModel, store_as
from pydantic import Field

from bo_eval.core import Session

T = TypeVar("T")


class BOState(StoreModel):
    session: Session = Field(default_factory=lambda: Session(env_name=""))


def session() -> Session:
    return store_as(BOState).session


def use_session(fn: Callable[[Session], T]) -> T:
    """Apply fn to the sample's Session, persist it, and surface ValueErrors to the model as ToolErrors."""
    st = store_as(BOState)
    s = st.session
    try:
        out = fn(s)
    except ValueError as e:
        raise ToolError(str(e))
    st.session = s
    return out
