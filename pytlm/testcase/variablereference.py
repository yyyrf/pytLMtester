#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides a base implementation of a variable in a test case."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

from pynguin.utils import type_utils
from pynguin.utils.type_utils import is_type_unknown

if TYPE_CHECKING:
    import pynguin.testcase.testcase as tc
    import pynguin.utils.generic.genericaccessibleobject as gao
    import pynguin.utils.namingscope as ns


class Reference(metaclass=ABCMeta):
    """Represents something that can be referenced in a test case.

    For example:
        foo_0 = Foo()
        int_0 = 42
        foo_0.bar = int_0

    Here, foo_0, int_0 and foo_0.bar are references.
    """

    def __init__(self, tp_: type | None) -> None:
        self._type = tp_

    @property
    def type(self) -> type | None:
        """Provides the type of this reference.

        Returns:
            The type of this reference
        """
        return self._type

    def is_primitive(self) -> bool:
        """Does this variable reference represent a primitive type.

        Returns:
            True if the variable is a primitive
        """
        return type_utils.is_primitive_type(self._type)

    def is_none_type(self) -> bool:
        """Is this variable reference of type none, i.e. it does not return anything.

        Returns:
            True if this variable is a none type
        """
        return type_utils.is_none_type(self._type)

    def is_type_unknown(self) -> bool:
        """Is the type of this variable unknown?

        Returns:
            True if this variable has unknown type
        """
        return is_type_unknown(self._type)

    @abstractmethod
    def get_names(
        self,
        variable_names: ns.AbstractNamingScope,
        module_names: ns.AbstractNamingScope,
    ) -> list[str]:
        """Get the names involved when addressing this reference.

        Args:
            variable_names: Mapping for variable names.
            module_names: Mapping for modules.

        Returns:
            A list of the elements in the reference names, i.e.,
            resolving a reference that represents "module_0.Foo.bar" gives
            ["module_0", "Foo", "bar"]
        """

    @abstractmethod
    def clone(self, memo: dict[VariableReference, VariableReference]) -> Reference:
        """Clone this reference.

        Args:
            memo: A mapping from variables in this test case to their corresponding
                variable in the new test case.

        Returns:
            A clone of this reference.
        """

    @abstractmethod
    def structural_eq(
        self, other: Any, memo: dict[VariableReference, VariableReference]
    ) -> bool:
        """Compare if this reference is the same as the other.

        Args:
            other: The variable to compare
            memo: A mapping from variables in this test case to their corresponding
                variable in the compared test case.

        Returns:
            True, iff this variable is the same as the other and points to the same
            location.
        """

    @abstractmethod
    def structural_hash(self) -> int:
        """Required for structural_eq to work.

        Returns:
            A hash value.
        """

    @abstractmethod
    def get_variable_reference(self) -> VariableReference | None:
        """Provide the variable reference used in this reference

        Returns: The variable reference used here, if any.
        """

    @abstractmethod
    def replace_variable_reference(
        self, old: VariableReference, new: VariableReference
    ) -> None:
        """Replace the old variable with the new variable.

        Args:
            old: The old variable reference
            new: The new variable reference
        """


class VariableReference(Reference):
    """A reference to a variable declared in a test case.

    For example:
        int_0 = 5

    Note that this class does NOT implement eq/hash because we use object identity
    to check for equality. The other reference do implement eq/hash.
    """

    def __init__(self, test_case: tc.TestCase, tp_: type | None):
        super().__init__(tp_)
        self._test_case = test_case
        self._distance = 0

    @property
    def test_case(self) -> tc.TestCase:
        """Provides the test case in which this variable reference is used.

        Returns:
            The containing test case
        """
        return self._test_case

    @property
    def distance(self) -> int:
        """Distance metric used to select variables for mutation based on how close
        they are to the subject under test.

        Returns:
            The distance value
        """
        return self._distance

    @distance.setter
    def distance(self, distance: int) -> None:
        """Set the distance metric.

        Args:
            distance: The new distance value
        """
        self._distance = distance

    def get_names(
        self,
        variable_names: ns.AbstractNamingScope,
        module_names: ns.AbstractNamingScope,
    ) -> list[str]:
        return [variable_names.get_name(self)]

    def clone(
        self, memo: dict[VariableReference, VariableReference]
    ) -> VariableReference:
        return memo[self]

    def structural_eq(
        self, other: Any, memo: dict[VariableReference, VariableReference]
    ) -> bool:
        if not isinstance(other, VariableReference):
            return False
        return self._type == other._type and memo[self] == other

    def structural_hash(self) -> int:
        return 31 * 17 + hash(self._type)

    def get_statement_position(self) -> int:
        """Provides the position of the statement which defines this variable reference
        in the test case.

        Raises:
            Exception: if the statement is not found in the test case

        Returns:
            The position  # noqa: DAR202
        """
        for idx, stmt in enumerate(self._test_case.statements):
            if stmt.ret_val == self:
                return idx
        raise Exception(
            "Variable reference is not declared in the test case in which it is used"
        )

    def get_variable_reference(self) -> VariableReference | None:
        return self

    def replace_variable_reference(
        self, old: VariableReference, new: VariableReference
    ) -> None:
        # We can't replace ourselves.
        return


class FieldReference(Reference):
    """A reference to a non-static field."""

    def __init__(self, source: Reference, field: gao.GenericField):
        super().__init__(field.generated_type())
        self._source = source
        self._field = field

    @property
    def source(self) -> Reference:
        """Provide the source.

        Returns:
            The source.
        """
        return self._source

    @property
    def field(self) -> gao.GenericField:
        """Provide the field.

        Returns:
            The field
        """
        return self._field

    def get_names(
        self,
        variable_names: ns.AbstractNamingScope,
        module_names: ns.AbstractNamingScope,
    ) -> list[str]:
        lst = self._source.get_names(variable_names, module_names)
        lst.append(self._field.field)
        return lst

    def clone(self, memo: dict[VariableReference, VariableReference]) -> FieldReference:
        return FieldReference(self._source.clone(memo), self._field)

    def structural_eq(
        self, other: Any, memo: dict[VariableReference, VariableReference]
    ) -> bool:
        if not isinstance(other, FieldReference):
            return False
        return self._field == other._field and self._source.structural_eq(
            other._source, memo
        )

    def structural_hash(self) -> int:
        return hash((self._field, self._source.structural_hash()))

    def __eq__(self, other):
        if not isinstance(other, FieldReference):
            return False
        return self._field == other._field and self._source == other._source

    def __hash__(self):
        return hash((self._field, self._source))

    def get_variable_reference(self) -> VariableReference | None:
        return self._source.get_variable_reference()

    def replace_variable_reference(
        self, old: VariableReference, new: VariableReference
    ) -> None:
        if self._source == old:
            self._source = new
        else:
            self._source.replace_variable_reference(old, new)


class StaticFieldReference(Reference):
    """A reference to a static field of a class."""

    def __init__(self, field: gao.GenericStaticField):
        super().__init__(field.generated_type())
        self._field = field

    @property
    def field(self) -> gao.GenericStaticField:
        """Provide the field.

        Returns:
            The field
        """
        return self._field

    def get_names(
        self,
        variable_names: ns.AbstractNamingScope,
        module_names: ns.AbstractNamingScope,
    ) -> list[str]:
        assert self._field.owner is not None
        return [
            module_names.get_name(self._field.owner.__module__),
            self._field.owner.__name__,
            self._field.field,
        ]

    def clone(
        self, memo: dict[VariableReference, VariableReference]
    ) -> StaticFieldReference:
        return StaticFieldReference(self._field)

    def structural_eq(
        self, other: Any, memo: dict[VariableReference, VariableReference]
    ) -> bool:
        if not isinstance(other, StaticFieldReference):
            return False
        return self._field == other._field

    def structural_hash(self) -> int:
        return hash(self._field)

    def __eq__(self, other):
        return self.structural_eq(other, {})

    def __hash__(self):
        return self.structural_hash()

    def get_variable_reference(self) -> VariableReference | None:
        return None

    def replace_variable_reference(
        self, old: VariableReference, new: VariableReference
    ) -> None:
        return


class StaticModuleFieldReference(Reference):
    """A reference to a static module field."""

    # TODO(fk) combine with regular static field?

    def __init__(self, field: gao.GenericStaticModuleField):
        super().__init__(field.generated_type())
        self._field = field

    @property
    def field(self) -> gao.GenericStaticModuleField:
        """Provide the field.

        Returns:
            The field
        """
        return self._field

    def get_names(
        self,
        variable_names: ns.AbstractNamingScope,
        module_names: ns.AbstractNamingScope,
    ) -> list[str]:
        return [module_names.get_name(self._field.module), self._field.field]

    def clone(
        self, memo: dict[VariableReference, VariableReference]
    ) -> StaticModuleFieldReference:
        return StaticModuleFieldReference(self._field)

    def structural_eq(
        self, other: Any, memo: dict[VariableReference, VariableReference]
    ) -> bool:
        if not isinstance(other, StaticModuleFieldReference):
            return False
        return self._field == other._field

    def structural_hash(self) -> int:
        return hash(self._field)

    def __eq__(self, other):
        return self.structural_eq(other, {})

    def __hash__(self):
        return self.structural_hash()

    def get_variable_reference(self) -> VariableReference | None:
        return None

    def replace_variable_reference(
        self, old: VariableReference, new: VariableReference
    ) -> None:
        return
