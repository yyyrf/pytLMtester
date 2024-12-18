#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2020 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides classes for various bytecode instrumentations."""
from __future__ import annotations

import json
import logging
from types import CodeType
from typing import TYPE_CHECKING

from bytecode import BasicBlock, Bytecode, Compare, ControlFlowGraph, Instr

from pynguin.analyses.controlflow import CFG, ControlDependenceGraph
from pynguin.analyses.seeding import DynamicConstantSeeding
from pynguin.testcase.execution import (
    CodeObjectMetaData,
    ExecutionTracer,
    PredicateMetaData,
)

if TYPE_CHECKING:
    from pynguin.analyses.controlflow import ProgramGraphNode

CODE_OBJECT_ID_KEY = "code_object_id"


# pylint:disable=too-few-public-methods
class InstrumentationAdapter:
    """Abstract base class for bytecode instrumentation adapters.

    General notes:

    When calling a method on an object, the arguments have to be on top of the stack.
    In most cases, we need to rotate the items on the stack with ROT_THREE or ROT_FOUR
    to reorder the elements accordingly.

    A POP_TOP instruction is required after calling a method, because each method
    implicitly returns None.

    This class defines visit_* methods that are called from the
    InstrumentationTransformer. Each subclass should override the visit_* methods
    where it wants to do something.
    """

    # TODO(fk) make this more fine grained? e.g. visit_line, visit_compare etc.
    #  Or use sub visitors?

    def visit_entry_node(self, block: BasicBlock, code_object_id: int) -> None:
        """Called when we visit the entry node of a code object.

        Args:
            block: The basic block of the entry node.
            code_object_id: The code object id of the containing code object.
        """

    def visit_node(
        self,
        cfg: CFG,
        code_object_id: int,
        node: ProgramGraphNode,
        basic_block: BasicBlock,
    ) -> None:
        """Called for each non-artificial node, i.e., nodes that have a basic block

        Args:
            cfg: The control flow graph.
            code_object_id: The code object id of the containing code object.
            node: The node in the control flow graph.
            basic_block: The basic block associated with the node.
        """

    @staticmethod
    def _create_consecutive_blocks(
        bytecode_cfg: ControlFlowGraph, first: BasicBlock, amount: int
    ) -> tuple[BasicBlock, ...]:
        """Split the given basic block into more blocks.

        The blocks are consecutive in the list of basic blocks, e.g., to allow
        fall-through

        Args:
            bytecode_cfg: The control-flow graph
            first: The first basic block
            amount: The amount of consecutive blocks that should be created.

        Returns:
            A tuple of consecutive basic blocks
        """
        assert amount > 0, "Amount of created basic blocks must be positive."
        current: BasicBlock = first
        nodes: list[BasicBlock] = []
        # Can be any instruction, as it is discarded anyway.
        dummy_instruction = Instr("POP_TOP")
        for _ in range(amount):
            # Insert dummy instruction, which we can use to split off another block
            current.insert(0, dummy_instruction)
            current = bytecode_cfg.split_block(current, 1)
            nodes.append(current)

        # Move instructions back to first block.
        first.clear()
        first.extend(current)
        # Clear instructions in all created blocks.
        for node in nodes:
            node.clear()
        return tuple(nodes)


class InstrumentationTransformer:
    """Applies a given list of instrumentation adapters to code objects.

    This class is responsible for traversing all nested code objects and their
    basic blocks and requesting their instrumentation from the given adapters.

    Ideally we would want something like ASM with nested visitors where changes from
    different adapters don't affect each other, but that's a bit of overkill for now.
    """

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        tracer: ExecutionTracer,
        instrumentation_adapters: list[InstrumentationAdapter],
    ):
        self._instrumentation_adapters = instrumentation_adapters
        self._tracer = tracer

    def instrument_module(self, module_code: CodeType) -> CodeType:
        """Instrument the given code object of a module.

        Args:
            module_code: The code object of the module

        Returns:
            The instrumented code object of the module
        """
        for const in module_code.co_consts:
            if isinstance(const, ExecutionTracer):
                # Abort instrumentation, since we have already
                # instrumented this code object.
                assert False, "Tried to instrument already instrumented module."
        return self._instrument_code_recursive(module_code)

    def _instrument_code_recursive(
        self,
        code: CodeType,
        parent_code_object_id: int | None = None,
    ) -> CodeType:
        """Instrument the given Code Object recursively.

        Args:
            code: The code object that should be instrumented
            parent_code_object_id: The ID of the optional parent code object

        Returns:
            The instrumented code object
        """
        self._logger.debug("Instrumenting Code Object for %s", code.co_name)
        cfg = CFG.from_bytecode(Bytecode.from_code(code))
        cdg = ControlDependenceGraph.compute(cfg)
        code_object_id = self._tracer.register_code_object(
            CodeObjectMetaData(
                code_object=code,
                parent_code_object_id=parent_code_object_id,
                cfg=cfg,
                cdg=cdg,
            )
        )
        # Overwrite/Set docstring to carry tagging information, i.e.,
        # the code object id. Convert to JSON string because I'm not sure where this
        # value might be used in CPython.
        cfg.bytecode_cfg().docstring = json.dumps({CODE_OBJECT_ID_KEY: code_object_id})
        assert cfg.entry_node is not None, "Entry node cannot be None."
        real_entry_node = cfg.get_successors(cfg.entry_node).pop()  # Only one exists!
        assert real_entry_node.basic_block is not None, "Basic block cannot be None."
        for adapter in self._instrumentation_adapters:
            adapter.visit_entry_node(real_entry_node.basic_block, code_object_id)
        self._instrument_cfg(cfg, code_object_id)
        return self._instrument_inner_code_objects(
            cfg.bytecode_cfg().to_code(), code_object_id
        )

    def _instrument_inner_code_objects(
        self, code: CodeType, parent_code_object_id: int
    ) -> CodeType:
        """Apply the instrumentation to all constants of the given code object.

        Args:
            code: the Code Object that should be instrumented.
            parent_code_object_id: the id of the parent code object, if any.

        Returns:
            the code object whose constants were instrumented.
        """
        new_consts = []
        for const in code.co_consts:
            if isinstance(const, CodeType):
                # The const is an inner code object
                new_consts.append(
                    self._instrument_code_recursive(
                        const, parent_code_object_id=parent_code_object_id
                    )
                )
            else:
                new_consts.append(const)
        return code.replace(co_consts=tuple(new_consts))

    def _instrument_cfg(self, cfg: CFG, code_object_id: int) -> None:
        """Instrument the bytecode cfg associated with the given CFG.

        Args:
            cfg: The CFG that overlays the bytecode cfg.
            code_object_id: The id of the code object which contains this CFG.
        """
        for node in cfg.nodes:
            if node.is_artificial:
                # Artificial nodes don't have a basic block, so we don't need to
                # instrument anything.
                continue
            assert (
                node.basic_block is not None
            ), "Non artificial node does not have a basic block."
            for adapter in self._instrumentation_adapters:
                adapter.visit_node(cfg, code_object_id, node, node.basic_block)


class BranchCoverageInstrumentation(InstrumentationAdapter):
    """Instruments code objects to enable tracking branch distances and thus
    branch coverage."""

    # Conditional jump operations are the last operation within a basic block
    _JUMP_OP_POS = -1

    # If a conditional jump is based on a comparison, it has to be the second-to-last
    # instruction within the basic block.
    _COMPARE_OP_POS = -2

    _logger = logging.getLogger(__name__)

    def __init__(self, tracer: ExecutionTracer) -> None:
        self._tracer = tracer

    def visit_node(
        self,
        cfg: CFG,
        code_object_id: int,
        node: ProgramGraphNode,
        basic_block: BasicBlock,
    ) -> None:
        """Instrument a single node in the CFG.

        Currently, we only instrument conditional jumps and for loops.

        Args:
            cfg: The containing CFG.
            code_object_id: The containing Code Object
            node: The node that should be instrumented.
            basic_block: The basic block of the node that should be instrumented.
        """

        assert len(basic_block) > 0, "Empty basic block in CFG."
        maybe_jump: Instr = basic_block[self._JUMP_OP_POS]
        maybe_compare: Instr | None = (
            basic_block[self._COMPARE_OP_POS] if len(basic_block) > 1 else None
        )
        if isinstance(maybe_jump, Instr):
            predicate_id: int | None = None
            if maybe_jump.name == "FOR_ITER":
                predicate_id = self._instrument_for_loop(
                    cfg, node, basic_block, code_object_id
                )
            elif maybe_jump.is_cond_jump():
                predicate_id = self._instrument_cond_jump(
                    code_object_id,
                    maybe_compare,
                    maybe_jump,
                    basic_block,
                    node,
                )
            if predicate_id is not None:
                node.predicate_id = predicate_id

    def _instrument_cond_jump(
        self,
        code_object_id: int,
        maybe_compare: Instr | None,
        jump: Instr,
        block: BasicBlock,
        node: ProgramGraphNode,
    ) -> int:
        # pylint:disable=too-many-arguments
        """Instrument a conditional jump.

        If it is based on a prior comparison, we track
        the compared values, otherwise we just track the truthiness of the value on top
        of the stack.

        Args:
            code_object_id: The id of the containing Code Object.
            maybe_compare: The comparison operation, if any.
            jump: The jump operation.
            block: The containing basic block.
            node: The associated node from the CFG.

        Returns:
            The id that was assigned to the predicate.
        """
        if (
            maybe_compare is not None
            and isinstance(maybe_compare, Instr)
            and maybe_compare.name in ("COMPARE_OP", "IS_OP", "CONTAINS_OP")
        ):
            return self._instrument_compare_based_conditional_jump(
                block, code_object_id, node
            )
        if jump.name == "JUMP_IF_NOT_EXC_MATCH":
            return self._instrument_exception_based_conditional_jump(
                block, code_object_id, node
            )
        return self._instrument_bool_based_conditional_jump(block, code_object_id, node)

    def _instrument_bool_based_conditional_jump(
        self, block: BasicBlock, code_object_id: int, node: ProgramGraphNode
    ) -> int:
        """Instrument boolean-based conditional jumps.

        We add a call to the tracer which reports the value on which the conditional
        jump will be based.

        Args:
            block: The containing basic block.
            code_object_id: The id of the containing Code Object.
            node: The associated node from the CFG.

        Returns:
            The id assigned to the predicate.
        """
        lineno = block[self._JUMP_OP_POS].lineno
        predicate_id = self._tracer.register_predicate(
            PredicateMetaData(line_no=lineno, code_object_id=code_object_id, node=node)
        )
        # Insert instructions right before the conditional jump.
        # We duplicate the value on top of the stack and report
        # it to the tracer.
        block[self._JUMP_OP_POS : self._JUMP_OP_POS] = [
            Instr("DUP_TOP", lineno=lineno),
            Instr("LOAD_CONST", self._tracer, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                ExecutionTracer.executed_bool_predicate.__name__,
                lineno=lineno,
            ),
            Instr("ROT_THREE", lineno=lineno),
            Instr("ROT_THREE", lineno=lineno),
            Instr("LOAD_CONST", predicate_id, lineno=lineno),
            Instr("CALL_METHOD", 2, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]
        return predicate_id

    def _instrument_compare_based_conditional_jump(
        self, block: BasicBlock, code_object_id: int, node: ProgramGraphNode
    ) -> int:
        """Instrument compare-based conditional jumps.

        We add a call to the tracer which reports the values that will be used
        in the following comparison operation on which the conditional jump is based.

        Args:
            block: The containing basic block.
            code_object_id: The id of the containing Code Object.
            node: The associated node from the CFG.

        Raises:
            RuntimeError: If an unknown operation is encountered.

        Returns:
            The id assigned to the predicate.
        """
        lineno = block[self._JUMP_OP_POS].lineno
        predicate_id = self._tracer.register_predicate(
            PredicateMetaData(line_no=lineno, code_object_id=code_object_id, node=node)
        )
        operation = block[self._COMPARE_OP_POS]

        match operation.name:
            case "COMPARE_OP":
                compare = operation.arg
            case "IS_OP":
                # Beginning with 3.9, there are separate OPs for various comparisons.
                # Map them back to the old operations, so we can use the enum from the
                # bytecode library.
                compare = Compare.IS_NOT if operation.arg else Compare.IS
            case "CONTAINS_OP":
                compare = Compare.NOT_IN if operation.arg else Compare.IN
            case _:
                raise RuntimeError(f"Unknown comparison OP {operation}")

        # Insert instructions right before the comparison.
        # We duplicate the values on top of the stack and report
        # them to the tracer.
        block[self._COMPARE_OP_POS : self._COMPARE_OP_POS] = [
            Instr("DUP_TOP_TWO", lineno=lineno),
            Instr("LOAD_CONST", self._tracer, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                ExecutionTracer.executed_compare_predicate.__name__,
                lineno=lineno,
            ),
            Instr("ROT_FOUR", lineno=lineno),
            Instr("ROT_FOUR", lineno=lineno),
            Instr("LOAD_CONST", predicate_id, lineno=lineno),
            Instr("LOAD_CONST", compare, lineno=lineno),
            Instr("CALL_METHOD", 4, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]
        return predicate_id

    def _instrument_exception_based_conditional_jump(
        self, block: BasicBlock, code_object_id: int, node: ProgramGraphNode
    ) -> int:
        """Instrument exception-based conditional jumps.

        We add a call to the tracer which reports the values that will be used
        in the following exception matching case.

        Args:
            block: The containing basic block.
            code_object_id: The id of the containing Code Object.
            node: The associated node from the CFG.

        Returns:
            The id assigned to the predicate.
        """
        lineno = block[self._JUMP_OP_POS].lineno
        predicate_id = self._tracer.register_predicate(
            PredicateMetaData(line_no=lineno, code_object_id=code_object_id, node=node)
        )
        # Insert instructions right before the conditional jump.
        # We duplicate the values on top of the stack and report
        # them to the tracer.
        block[self._JUMP_OP_POS : self._JUMP_OP_POS] = [
            Instr("DUP_TOP_TWO", lineno=lineno),
            Instr("LOAD_CONST", self._tracer, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                ExecutionTracer.executed_exception_match.__name__,
                lineno=lineno,
            ),
            Instr("ROT_FOUR", lineno=lineno),
            Instr("ROT_FOUR", lineno=lineno),
            Instr("LOAD_CONST", predicate_id, lineno=lineno),
            Instr("CALL_METHOD", 3, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]
        return predicate_id

    def visit_entry_node(self, block: BasicBlock, code_object_id: int) -> None:
        """Add instructions at the beginning of the given basic block which inform
        the tracer, that the code object with the given id has been entered.

        Args:
            block: The entry basic block of a code object, i.e. the first basic block.
            code_object_id: The id that the tracer has assigned to the code object
                which contains the given basic block.
        """
        # Use line number of first instruction
        lineno = block[0].lineno
        # Insert instructions at the beginning.
        block[0:0] = [
            Instr("LOAD_CONST", self._tracer, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                ExecutionTracer.executed_code_object.__name__,
                lineno=lineno,
            ),
            Instr("LOAD_CONST", code_object_id, lineno=lineno),
            Instr("CALL_METHOD", 1, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]

    def _instrument_for_loop(
        self,
        cfg: CFG,
        node: ProgramGraphNode,
        basic_block: BasicBlock,
        code_object_id: int,
    ) -> int:
        """Transform the for loop whose header is defined in the given node.
        We only transform the underlying bytecode cfg, by partially unrolling the first
        iteration. For this, we add two basic blocks after the loop header:

        The first block is called, if the iterator on which the loop is based
        yields at least one element, in which case we report the boolean value True
        to the tracer, leave the yielded value of the iterator on top of the stack and
        jump to the regular body of the loop.

        The second block is called, if the iterator on which the loop is based
        does not yield an element, in which case we report the boolean value False
        to the tracer and jump to the exit instruction of the loop.

        The original loop header is changed such that it either falls through to the
        first block or jumps to the second, if no element is yielded.

        Since Python is a structured programming language, there can be no jumps
        directly into the loop that bypass the loop header (e.g., GOTO).
        Jumps which reach the loop header from outside the loop will still target
        the original loop header, so they don't need to be modified.

        Attention! These changes to the control flow are not reflected in the high level
        CFG, but only in the bytecode CFG.

        Args:
            cfg: The CFG that contains the loop
            node: The node which contains the header of the for loop.
            basic_block: The basic block of the node.
            code_object_id: The id of the containing Code Object.

        Returns:
            The ID of the instrumented predicate
        """
        for_instr = basic_block[self._JUMP_OP_POS]
        assert for_instr.name == "FOR_ITER"
        lineno = for_instr.lineno
        predicate_id = self._tracer.register_predicate(
            PredicateMetaData(line_no=lineno, code_object_id=code_object_id, node=node)
        )
        for_loop_exit = for_instr.arg
        for_loop_body = basic_block.next_block

        # pylint:disable=unbalanced-tuple-unpacking
        entered, not_entered = self._create_consecutive_blocks(
            cfg.bytecode_cfg(), basic_block, 2
        )
        for_instr.arg = not_entered

        entered.extend(
            [
                Instr("LOAD_CONST", self._tracer, lineno=lineno),
                Instr(
                    "LOAD_METHOD",
                    ExecutionTracer.executed_bool_predicate.__name__,
                    lineno=lineno,
                ),
                Instr("LOAD_CONST", True, lineno=lineno),
                Instr("LOAD_CONST", predicate_id, lineno=lineno),
                Instr("CALL_METHOD", 2, lineno=lineno),
                Instr("POP_TOP", lineno=lineno),
                Instr("JUMP_ABSOLUTE", for_loop_body, lineno=lineno),
            ]
        )

        not_entered.extend(
            [
                Instr("LOAD_CONST", self._tracer, lineno=lineno),
                Instr(
                    "LOAD_METHOD",
                    ExecutionTracer.executed_bool_predicate.__name__,
                    lineno=lineno,
                ),
                Instr("LOAD_CONST", False, lineno=lineno),
                Instr("LOAD_CONST", predicate_id, lineno=lineno),
                Instr("CALL_METHOD", 2, lineno=lineno),
                Instr("POP_TOP", lineno=lineno),
                Instr("JUMP_ABSOLUTE", for_loop_exit, lineno=lineno),
            ]
        )

        return predicate_id


# pylint:disable=too-few-public-methods
class LineCoverageInstrumentation(InstrumentationAdapter):
    """Instruments code objects to enable tracking of executed lines and thus
    line coverage."""

    _logger = logging.getLogger(__name__)

    def __init__(self, tracer: ExecutionTracer) -> None:
        self._tracer = tracer

    def visit_node(
        self,
        cfg: CFG,
        code_object_id: int,
        node: ProgramGraphNode,
        basic_block: BasicBlock,
    ) -> None:
        #  iterate over instructions after the fist one in BB,
        #  put new instructions in the block for each line
        file_name = cfg.bytecode_cfg().filename
        lineno = None
        instr_index = 0
        while instr_index < len(basic_block):
            if basic_block[instr_index].lineno != lineno:
                lineno = basic_block[instr_index].lineno
                line_id = self._tracer.register_line(code_object_id, file_name, lineno)
                instr_index += (  # increment by the amount of instructions inserted
                    self.instrument_line(basic_block, instr_index, line_id, lineno)
                )
            instr_index += 1

    def instrument_line(
        self, block: BasicBlock, instr_index: int, line_id: int, lineno: int
    ) -> int:
        """Instrument instructions of a new line.

        We add a call to the tracer which reports a line was executed.

        Args:
            block: The basic block containing the instrumented line.
            instr_index: the index of the instr
            line_id: The id of the line that is visited.
            lineno: The line number of the instrumented line.

        Returns:
            The number of instructions inserted into the block
        """
        inserted_instructions = [
            Instr("LOAD_CONST", self._tracer, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                self._tracer.track_line_visit.__name__,
                lineno=lineno,
            ),
            Instr("LOAD_CONST", line_id, lineno=lineno),
            Instr("CALL_METHOD", 1, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]
        # Insert instructions at the beginning.
        block[instr_index:instr_index] = inserted_instructions
        return len(inserted_instructions)


# pylint:disable=too-few-public-methods
class DynamicSeedingInstrumentation(InstrumentationAdapter):
    """Instruments code objects to enable dynamic constant seeding.

    Supported is collecting values of the types int, float and string.

    Instrumented are the common compare operations (==, !=, <, >, <=, >=) and the string
    methods contained in the STRING_FUNCTION_NAMES list. This means, if one of the
    above operations and methods is used in an if-conditional, corresponding values
    are added to the dynamic constant pool.

    The dynamic pool is implemented in the module constantseeding.py. The dynamicseeding
    module containes methods for managing the dynamic pool during the algorithm
    execution."""

    # Compare operations are only followed by one jump operation, hence they are on the
    # second to last position of the block.
    _COMPARE_OP_POS = -2

    #  If one of the considered string functions needing no argument is used in the if
    #  statement, it will be loaded in the third last position. After it comes the
    #  call of the method and the jump operation.
    _STRING_FUNC_POS = -3

    # If one of the considered string functions needing one argument is used in the if
    # statement, it will be loaded in the fourth last position. After it comes the
    # load of the argument, the call of the method and the jump
    # operation.
    _STRING_FUNC_POS_WITH_ARG = -4

    # A list containing the names of all string functions which are instrumented.
    _STRING_FUNCTION_NAMES = [
        "startswith",
        "endswith",
        "isalnum",
        "isalpha",
        "isdecimal",
        "isdigit",
        "isidentifier",
        "islower",
        "isnumeric",
        "isprintable",
        "isspace",
        "istitle",
        "isupper",
    ]

    _logger = logging.getLogger(__name__)

    def __init__(self, dynamic_constant_seeding: DynamicConstantSeeding):
        self._dynamic_constant_seeding = dynamic_constant_seeding

    def visit_node(
        self,
        cfg: CFG,
        code_object_id: int,
        node: ProgramGraphNode,
        basic_block: BasicBlock,
    ) -> None:
        assert len(basic_block) > 0, "Empty basic block in CFG."
        maybe_compare: Instr | None = (
            basic_block[self._COMPARE_OP_POS] if len(basic_block) > 1 else None
        )
        maybe_string_func: Instr | None = (
            basic_block[self._STRING_FUNC_POS] if len(basic_block) > 2 else None
        )
        maybe_string_func_with_arg: Instr | None = (
            basic_block[self._STRING_FUNC_POS_WITH_ARG]
            if len(basic_block) > 3
            else None
        )
        if isinstance(maybe_compare, Instr) and maybe_compare.name == "COMPARE_OP":
            self._instrument_compare_op(basic_block)
        if (
            isinstance(maybe_string_func, Instr)
            and maybe_string_func.name == "LOAD_METHOD"
            and maybe_string_func.arg in self._STRING_FUNCTION_NAMES
        ):
            self._instrument_string_func(basic_block, maybe_string_func.arg)
        if (
            isinstance(maybe_string_func_with_arg, Instr)
            and maybe_string_func_with_arg.name == "LOAD_METHOD"
            and maybe_string_func_with_arg.arg in self._STRING_FUNCTION_NAMES
        ):
            self._instrument_string_func(basic_block, maybe_string_func_with_arg.arg)

    def _instrument_startswith_function(self, block: BasicBlock) -> None:
        """Instruments the startswith function in bytecode. Stores for the expression
          'string1.startswith(string2)' the
           value 'string2 + string1' in the _dynamic_pool.

        Args:
            block: The basic block where the new instructions are inserted.
        """
        insert_pos = self._STRING_FUNC_POS_WITH_ARG + 2
        lineno = block[insert_pos].lineno
        block[insert_pos:insert_pos] = [
            Instr("DUP_TOP_TWO", lineno=lineno),
            Instr("ROT_TWO", lineno=lineno),
            Instr("BINARY_ADD", lineno=lineno),
            Instr("LOAD_CONST", self._dynamic_constant_seeding, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                self._dynamic_constant_seeding.add_value.__name__,
                lineno=lineno,
            ),
            Instr("ROT_THREE", lineno=lineno),
            Instr("ROT_THREE", lineno=lineno),
            Instr("CALL_METHOD", 1, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]
        self._logger.info("Instrumented startswith function")

    def _instrument_endswith_function(self, block: BasicBlock) -> None:
        """Instruments the endswith function in bytecode. Stores for the expression
         'string1.startswith(string2)' the
           value 'string1 + string2' in the _dynamic_pool.

        Args:
            block: The basic block where the new instructions are inserted.
        """
        insert_pos = self._STRING_FUNC_POS_WITH_ARG + 2
        lineno = block[insert_pos].lineno
        block[insert_pos:insert_pos] = [
            Instr("DUP_TOP_TWO", lineno=lineno),
            Instr("BINARY_ADD", lineno=lineno),
            Instr("LOAD_CONST", self._dynamic_constant_seeding, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                DynamicConstantSeeding.add_value.__name__,
                lineno=lineno,
            ),
            Instr("ROT_THREE", lineno=lineno),
            Instr("ROT_THREE", lineno=lineno),
            Instr("CALL_METHOD", 1, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]
        self._logger.info("Instrumented endswith function")

    def _instrument_string_function_without_arg(
        self, block: BasicBlock, function_name: str
    ) -> None:
        """Instruments the isalnum function in bytecode.

        Args:
            block: The basic block where the new instructions are inserted.
            function_name: The name of the function
        """
        insert_pos = self._STRING_FUNC_POS_WITH_ARG + 2
        lineno = block[insert_pos].lineno
        block[insert_pos:insert_pos] = [
            Instr("DUP_TOP", lineno=lineno),
            Instr("LOAD_CONST", self._dynamic_constant_seeding, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                DynamicConstantSeeding.add_value_for_strings.__name__,
                lineno=lineno,
            ),
            Instr("ROT_THREE", lineno=lineno),
            Instr("ROT_THREE", lineno=lineno),
            Instr("LOAD_CONST", function_name, lineno=lineno),
            Instr("CALL_METHOD", 2, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]
        self._logger.info("Instrumented string function")

    def _instrument_string_func(self, block: BasicBlock, function_name: str) -> None:
        """Calls the corresponding instrumentation method for the given function_name.

        Args:
            block: The block to instrument.
            function_name: The name of the function for which the method will be called.

        """
        if function_name == "startswith":
            self._instrument_startswith_function(block)
        elif function_name == "endswith":
            self._instrument_endswith_function(block)
        else:
            self._instrument_string_function_without_arg(block, function_name)

    def _instrument_compare_op(self, block: BasicBlock) -> None:
        """Instruments the compare operations in bytecode. Stores the values extracted
         at runtime.

        Args:
            block: The containing basic block.
        """
        lineno = block[self._COMPARE_OP_POS].lineno
        block[self._COMPARE_OP_POS : self._COMPARE_OP_POS] = [
            Instr("DUP_TOP_TWO", lineno=lineno),
            Instr("LOAD_CONST", self._dynamic_constant_seeding, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                DynamicConstantSeeding.add_value.__name__,
                lineno=lineno,
            ),
            Instr("ROT_THREE", lineno=lineno),
            Instr("ROT_THREE", lineno=lineno),
            Instr("CALL_METHOD", 1, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
            Instr("LOAD_CONST", self._dynamic_constant_seeding, lineno=lineno),
            Instr(
                "LOAD_METHOD",
                DynamicConstantSeeding.add_value.__name__,
                lineno=lineno,
            ),
            Instr("ROT_THREE", lineno=lineno),
            Instr("ROT_THREE", lineno=lineno),
            Instr("CALL_METHOD", 1, lineno=lineno),
            Instr("POP_TOP", lineno=lineno),
        ]
        self._logger.debug("Instrumented compare_op")
