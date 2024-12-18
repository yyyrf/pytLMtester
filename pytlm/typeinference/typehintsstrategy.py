#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides a strategy implementation that uses type hints."""
from __future__ import annotations

import inspect
import typing
from typing import Callable

from pynguin.typeinference.strategy import InferredSignature, TypeInferenceStrategy
from pynguin.utils.type_utils import wrap_var_param_type


# pylint: disable=too-few-public-methods
class TypeHintsInferenceStrategy(TypeInferenceStrategy):
    """A type inference strategy that simply parses the type hints.

    For classes it inspects the `__init__` method and uses its parameters.
    """

    def infer_type_info(self, method: Callable) -> InferredSignature:
        if inspect.isclass(method) and hasattr(method, "__init__"):
            return self._infer_type_info_for_callable(getattr(method, "__init__"))
        return self._infer_type_info_for_callable(method)

    @staticmethod
    def _infer_type_info_for_callable(method: Callable) -> InferredSignature:
        try:
            signature = inspect.signature(method)
        except (ValueError, TypeError):
            # If inspect. fails to give us a signature, just assume we have none.
            # This should not happen for functions in the main module under test
            signature = inspect.Signature()

        parameters: dict[str, type | None] = {}
        try:
            hints = typing.get_type_hints(method)
        except NameError:
            # get_type_hints can fail to resolve types that include forward references;
            # if this happens, just pretend we got no type hints rather than failing
            # the entire run
            hints = {}
        for param_name in signature.parameters:
            if param_name == "self":
                continue
            hint = hints.get(param_name, None)
            hint = wrap_var_param_type(hint, signature.parameters[param_name].kind)
            parameters[param_name] = hint

        return_type: type | None = hints.get("return", None)

        return InferredSignature(
            signature=signature, parameters=parameters, return_type=return_type
        )
