"""Lazy construction of usage-validated auxiliary OpenAI clients."""

from typing import Any, TYPE_CHECKING

from agent.process_bootstrap import _OpenAIProxy

if TYPE_CHECKING:
    from openai import OpenAI


class _AuxiliaryOpenAIProxy(_OpenAIProxy):
    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> "OpenAI":
        from agent.auxiliary_request_clients import openai_request_client_classes

        return openai_request_client_classes()[0](*args, **kwargs)
