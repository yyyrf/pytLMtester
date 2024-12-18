#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides an inference strategy for types."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from inspect import Parameter, Signature
from typing import Callable


@dataclass
class InferredSignature:
    """Encapsulates the types inferred for a method.

    The fields contain the following:
    - `signature`: Holds an `inspect.Signature` object as generated from the
      `inspect.signature` function
    - `parameters`: A dictionary mapping a parameter name to its type, if any.
    - `return_type`: The return type of a method, if any.

    The semantics of the `parameters` and `return_type` value for `None` is given as
    follows: the value `None` means that we do not yet know anything about this type,
    the value `NoneType` means that this parameter or return type is of type `None`,
    i.e., there is not parameter or return value.

    Consider the following example:
    - `def foo()` with `return_type = None` means we do not know what the return type is
    - `def bar() -> None` with `return_type = type(None) = NoneType` means that the
      function does not return anything.

    The types shall not be updated directly!  One is supposed to use the methods
    `update_parameter_type(parameter_name: str, parameter_type: Optional[type])` and
    `update_return_type(return_type: Optional[type])` to update the parameter or return
    type, respectively.  These methods will also adjust the value of the `signature`
    field by generating a new `inspect.Signature` instance accordingly.
    """

    signature: Signature
    parameters: dict[str, type | None] = field(default_factory=dict)
    return_type: type | None = None

    def update_parameter_type(
        self, parameter_name: str, parameter_type: type | None
    ) -> None:
        """Updates the type of one parameter.

        Args:
            parameter_name: The name of the parameter
            parameter_type: The new type of the parameter
        """
        assert parameter_name in self.parameters
        self.parameters[parameter_name] = parameter_type
        self._update_signature_parameter(parameter_name, parameter_type)

    def update_return_type(self, return_type: type | None) -> None:
        """Updates the return type

        Args:
            return_type: The new return type
        """
        self.return_type = return_type
        self._update_signature_return_type(return_type)

    def _update_signature_parameter(
        self,
        parameter_name: str,
        parameter_type: type | None,
    ):
        current_parameter: Parameter | None = self.signature.parameters.get(
            parameter_name
        )
        assert current_parameter is not None, "Cannot happen due to previous check"
        new_parameter = current_parameter.replace(annotation=parameter_type)
        new_parameters = [
            new_parameter if key == parameter_name else value
            for key, value in self.signature.parameters.items()
        ]
        new_signature = self.signature.replace(parameters=new_parameters)
        self.signature = new_signature

    def _update_signature_return_type(self, return_type: type | None):
        new_signature = self.signature.replace(return_annotation=return_type)
        self.signature = new_signature


# pylint: disable=too-few-public-methods
class TypeInferenceStrategy(metaclass=ABCMeta):
    """Provides an abstract base class for inference strategies for types."""

    @abstractmethod
    def infer_type_info(self, method: Callable) -> InferredSignature:
        """Infers the type information for a callable.

        Args:
            method: The callable we try to infer type information for

        Returns:
            A MethodType object with the inference results  # noqa: DAR202
        """
