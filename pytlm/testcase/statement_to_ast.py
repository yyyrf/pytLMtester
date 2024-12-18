#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides a visitor that transforms statements to ast"""
from __future__ import annotations

import ast
from inspect import Parameter
from typing import TYPE_CHECKING, Any, cast

import pynguin.utils.ast_util as au
from pynguin.testcase.statement import ASTAssignStatement, StatementVisitor
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
)

if TYPE_CHECKING:
    import pynguin.utils.namingscope as ns
    from pynguin.testcase.statement import (
        AssignmentStatement,
        BooleanPrimitiveStatement,
        BytesPrimitiveStatement,
        ConstructorStatement,
        DictStatement,
        EnumPrimitiveStatement,
        FieldStatement,
        FloatPrimitiveStatement,
        FunctionStatement,
        IntPrimitiveStatement,
        ListStatement,
        MethodStatement,
        NoneStatement,
        ParametrizedStatement,
        PrimitiveStatement,
        SetStatement,
        StringPrimitiveStatement,
        TupleStatement,
    )


class StatementToAstVisitor(StatementVisitor):
    """Visitor that transforms statements into a list of AST nodes."""

    def __init__(
        self,
        module_aliases: ns.AbstractNamingScope,
        variable_names: ns.AbstractNamingScope,
        wrap_nodes: bool = False,
    ) -> None:
        """Creates a new transformation visitor that transforms our internal
        statements to Python AST nodes.

        Args:
            module_aliases: A naming scope for module alias names
            variable_names: A naming scope for variable names
            wrap_nodes: If True, wrap the create AST nodes in a try-except block
        """
        self._ast_nodes: list[ast.stmt] = []
        self._variable_names = variable_names
        self._module_aliases = module_aliases
        self._wrap_nodes = wrap_nodes

    def append_nodes(self, statements: list[ast.stmt]) -> None:
        """Add additional nodes to the already generated nodes.

        Args:
            statements: the ast statements that are appended.

        """
        # TODO(fk) cleaner solution with nested visitors?
        self._ast_nodes.extend(statements)

    @property
    def ast_nodes(self) -> list[ast.stmt]:
        """Get the list of generated AST nodes.

        In case the `wrap_nodes` property was set, the nodes will be wrapped in
        ```
        try:
            [nodes]
        except BaseException:
            pass
        ```

        Returns:
            A list of AST nodes
        """
        if self._wrap_nodes:
            nodes: list[ast.stmt] = [
                ast.Try(
                    body=self._ast_nodes,
                    handlers=[
                        ast.ExceptHandler(
                            body=[ast.Pass()],
                            name=None,
                            type=ast.Name(ctx=ast.Load(), id="BaseException"),
                        )
                    ],
                    orelse=[],
                    finalbody=[],
                )
            ]
            return nodes
        return self._ast_nodes

    def visit_int_primitive_statement(self, stmt: IntPrimitiveStatement) -> None:
        self._ast_nodes.append(self._create_constant(stmt))

    def visit_float_primitive_statement(self, stmt: FloatPrimitiveStatement) -> None:
        self._ast_nodes.append(self._create_constant(stmt))

    def visit_string_primitive_statement(self, stmt: StringPrimitiveStatement) -> None:
        self._ast_nodes.append(self._create_constant(stmt))

    def visit_bytes_primitive_statement(self, stmt: BytesPrimitiveStatement) -> None:
        self._ast_nodes.append(self._create_constant(stmt))

    def visit_boolean_primitive_statement(
        self, stmt: BooleanPrimitiveStatement
    ) -> None:
        self._ast_nodes.append(self._create_constant(stmt))

    def visit_enum_statement(self, stmt: EnumPrimitiveStatement) -> None:
        owner = stmt.accessible_object().owner
        assert owner
        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=ast.Attribute(
                    value=ast.Attribute(
                        value=self._create_module_alias(owner.__module__),
                        attr=owner.__name__,
                        ctx=ast.Load(),
                    ),
                    attr=stmt.value_name,
                    ctx=ast.Load(),
                ),
            )
        )

    def visit_none_statement(self, stmt: NoneStatement) -> None:
        self._ast_nodes.append(self._create_constant(stmt))

    def visit_constructor_statement(self, stmt: ConstructorStatement) -> None:
        owner = stmt.accessible_object().owner
        assert owner
        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=ast.Call(
                    func=ast.Attribute(
                        attr=owner.__name__,
                        ctx=ast.Load(),
                        value=self._create_module_alias(owner.__module__),
                    ),
                    args=self._create_args(stmt),
                    keywords=self._create_kw_args(stmt),
                ),
            )
        )

    def visit_method_statement(self, stmt: MethodStatement) -> None:
        call = ast.Call(
            func=ast.Attribute(
                attr=stmt.accessible_object().callable.__name__,
                ctx=ast.Load(),
                value=au.create_full_name(
                    self._variable_names, self._module_aliases, stmt.callee, True
                ),
            ),
            args=self._create_args(stmt),
            keywords=self._create_kw_args(stmt),
        )
        if stmt.ret_val.is_none_type():
            node: ast.stmt = ast.Expr(value=call)
        else:
            node = ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=call,
            )
        self._ast_nodes.append(node)

    def visit_function_statement(self, stmt: FunctionStatement) -> None:
        call = ast.Call(
            func=ast.Attribute(
                attr=stmt.accessible_object().callable.__name__,
                ctx=ast.Load(),
                value=self._create_module_alias(
                    stmt.accessible_object().callable.__module__
                ),
            ),
            args=self._create_args(stmt),
            keywords=self._create_kw_args(stmt),
        )
        if stmt.ret_val.is_none_type():
            node: ast.stmt = ast.Expr(value=call)
        else:
            node = ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=call,
            )
        self._ast_nodes.append(node)

    def visit_field_statement(self, stmt: FieldStatement) -> None:
        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    ast.Name(
                        id=self._variable_names.get_name(stmt.ret_val),
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Attribute(
                    attr=stmt.field.field,
                    ctx=ast.Load(),
                    value=au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.source, True
                    ),
                ),
            )
        )

    def visit_assignment_statement(self, stmt: AssignmentStatement) -> None:
        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.lhs, False
                    )
                ],
                value=au.create_full_name(
                    self._variable_names, self._module_aliases, stmt.rhs, True
                ),
            )
        )

    def visit_list_statement(self, stmt: ListStatement) -> None:
        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=ast.List(
                    elts=[
                        au.create_full_name(
                            self._variable_names, self._module_aliases, x, True
                        )
                        for x in stmt.elements
                    ],
                    ctx=ast.Load(),
                ),
            )
        )

    def visit_set_statement(self, stmt: SetStatement) -> None:
        # There is no literal for empty sets, so we have to write "set()"
        inner: Any
        if len(stmt.elements) == 0:
            inner = ast.Call(
                func=ast.Name(id="set", ctx=ast.Load()), args=[], keywords=[]
            )
        else:
            inner = ast.Set(
                elts=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, x, True
                    )
                    for x in stmt.elements
                ],
                ctx=ast.Load(),
            )

        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=inner,
            )
        )

    def visit_tuple_statement(self, stmt: TupleStatement) -> None:
        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=ast.Tuple(
                    elts=[
                        au.create_full_name(
                            self._variable_names, self._module_aliases, x, True
                        )
                        for x in stmt.elements
                    ],
                    ctx=ast.Load(),
                ),
            )
        )

    def visit_dict_statement(self, stmt: DictStatement) -> None:
        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=ast.Dict(
                    keys=[
                        au.create_full_name(
                            self._variable_names, self._module_aliases, x[0], True
                        )
                        for x in stmt.elements
                    ],
                    values=[
                        au.create_full_name(
                            self._variable_names, self._module_aliases, x[1], True
                        )
                        for x in stmt.elements
                    ],
                ),
            )
        )

    def visit_ast_assign_statement(self, stmt: ASTAssignStatement) -> None:
        self._ast_nodes.append(
            ast.Assign(
                targets=[
                    au.create_full_name(
                        self._variable_names, self._module_aliases, stmt.ret_val, False
                    )
                ],
                value=stmt.get_rhs_as_normal_ast(
                    lambda x: au.create_full_name(
                        self._variable_names, self._module_aliases, x, True
                    )
                ),
            )
        )

    def _create_constant(self, stmt: PrimitiveStatement) -> ast.stmt:
        """All primitive values are constants.

        Args:
            stmt: The primitive statement

        Returns:
            The matching AST statement
        """
        return ast.Assign(
            targets=[
                au.create_full_name(
                    self._variable_names, self._module_aliases, stmt.ret_val, False
                )
            ],
            value=ast.Constant(value=stmt.value),
        )

    def _create_args(self, stmt: ParametrizedStatement) -> list[ast.expr]:
        """Creates the positional arguments, i.e., POSITIONAL_ONLY,
        POSITIONAL_OR_KEYWORD and VAR_POSITIONAL.

        Args:
            stmt: The parameterised statement

        Returns:
            A list of AST statements
        """
        args: list[ast.expr] = []
        gen_callable: GenericCallableAccessibleObject = cast(
            GenericCallableAccessibleObject, stmt.accessible_object()
        )
        for param_name in gen_callable.inferred_signature.parameters:
            if param_name in stmt.args:
                param_kind = gen_callable.inferred_signature.signature.parameters[
                    param_name
                ].kind
                if param_kind in (
                    Parameter.POSITIONAL_ONLY,
                    Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    args.append(
                        au.create_full_name(
                            self._variable_names,
                            self._module_aliases,
                            stmt.args[param_name],
                            True,
                        )
                    )
                elif param_kind == Parameter.VAR_POSITIONAL:
                    # Append *args, if necessary.
                    args.append(
                        ast.Starred(
                            value=au.create_full_name(
                                self._variable_names,
                                self._module_aliases,
                                stmt.args[param_name],
                                True,
                            ),
                            ctx=ast.Load(),
                        )
                    )
        return args

    def _create_kw_args(self, stmt: ParametrizedStatement) -> list[ast.keyword]:
        """Creates the keyword arguments, i.e., KEYWORD_ONLY or VAR_KEYWORD.

        Args:
            stmt: The parameterised statement

        Returns:
            A list of AST statements
        """
        kwargs = []
        gen_callable: GenericCallableAccessibleObject = cast(
            GenericCallableAccessibleObject, stmt.accessible_object()
        )
        for param_name in gen_callable.inferred_signature.parameters:
            if param_name in stmt.args:
                param_kind = gen_callable.inferred_signature.signature.parameters[
                    param_name
                ].kind
                if param_kind == Parameter.KEYWORD_ONLY:
                    kwargs.append(
                        ast.keyword(
                            arg=param_name,
                            value=au.create_full_name(
                                self._variable_names,
                                self._module_aliases,
                                stmt.args[param_name],
                                True,
                            ),
                        )
                    )
                elif param_kind == Parameter.VAR_KEYWORD:
                    # Append **kwargs, if necessary.
                    kwargs.append(
                        ast.keyword(
                            arg=None,
                            value=au.create_full_name(
                                self._variable_names,
                                self._module_aliases,
                                stmt.args[param_name],
                                True,
                            ),
                        )
                    )
        return kwargs

    def _create_module_alias(self, module_name) -> ast.Name:
        """Create a name node for a module alias.

        Args:
            module_name: The name of the module

        Returns:
            An AST statement
        """
        return ast.Name(id=self._module_aliases.get_name(module_name), ctx=ast.Load())


class DuckStatementToAstVisitor(StatementToAstVisitor):
    """A visitor"""
