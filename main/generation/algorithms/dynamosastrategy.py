#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
"""Provides the DynaMOSA test-generation strategy."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, cast, Set, List

import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from ordered_set import OrderedSet

import pynguin.configuration as config
import pynguin.coverage.branchgoals as bg
import pynguin.utils.statistics.statistics as stat
import pynguin.testcase.testcase as tc
from pynguin.analyses.seeding import languagemodelseeding
from pynguin.ga.operators.ranking.crowdingdistance import (
    fast_epsilon_dominance_assignment,
)
from pynguin.generation.algorithms.abstractmosastrategy import AbstractMOSATestStrategy
from pynguin.generation.export.pytestexporter import PyTestExporter
from pynguin.generation.stoppingconditions.stoppingcondition import MaxSearchTimeStoppingCondition
from pynguin.testcase.statement import ConstructorStatement, FunctionStatement, MethodStatement, ASTAssignStatement
from pynguin.utils import randomness
from pynguin.utils.exceptions import ConstructionFailedException
from pynguin.utils.statistics.runtimevariable import RuntimeVariable


import pynguin.ga.computations as ff
import pynguin.ga.testcasechromosome as tcc
import pynguin.ga.testsuitechromosome as tsc
from pynguin.generation.algorithms.archive import CoverageArchive
from pynguin.testcase.execution import KnownData


logger = logging.getLogger(__name__)

class DynaMOSATestStrategy(AbstractMOSATestStrategy):
    """Implements the Dynamic Many-Objective Sorting Algorithm DynaMOSA."""

    _logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        super().__init__()
        self._goals_manager: _GoalsManager
        self._num_pytlm_tests_added = 0
        self._num_mutant_pytlm_tests_added = 0
        self._num_added_tests_needed_expansion = 0
        self._num_added_tests_needed_uninterp = 0
        self._num_added_tests_needed_calls = 0
        self._plateau_len = config.configuration.pytlm.max_plateau_len

    def _log_num_pytlm_tests_added(self):
        scs = [
            sc
            for sc in self.stopping_conditions
            if isinstance(sc, MaxSearchTimeStoppingCondition)
        ]
        report_dir = config.configuration.statistics_output.report_dir
        if len(scs) > 0 and report_dir != "pynguin-report":
            search_time: MaxSearchTimeStoppingCondition = scs[0]
            with open(
                os.path.join(report_dir, "pytlm_timeline.csv"),
                "a+",
                encoding="UTF-8",
            ) as log_file:
                log_file.write(
                    f"{search_time.current_value()},{self._num_pytlm_tests_added}\n"
                )


    def _register_added_testcase(
        self, test_case: tc.TestCase, was_mutant: bool
    ) -> None:
        """Register that test_case was a test case generated during the targeted
        LLM generation phase, and any additional statistics we're tracking.

        注册test_case是在目标LLM生成阶段生成的测试用例，以及我们正在跟踪的任何其他统计数据。

        Args:
            test_case: the test case to register  参数为要注册的测试用例
        """
        self._num_pytlm_tests_added += 1
        if was_mutant:
            self._num_mutant_pytlm_tests_added += 1
        exporter = PyTestExporter(wrap_code=False)
        logger.info(
            "New population test case:\n %s",
            exporter.export_sequences_to_str([test_case]),
        )
        if any(
            var in config.configuration.statistics_output.output_variables
            for var in [
                RuntimeVariable.LLMNeededExpansion,
                RuntimeVariable.LLMNeededUninterpretedCallsOnly,
                RuntimeVariable.LLMNeededUninterpreted,
            ]
        ):
            needed_expansion = False
            needed_calls = False
            needed_uninterp = False
            for stmt in test_case.statements:
                if isinstance(
                    stmt, (ConstructorStatement, FunctionStatement, MethodStatement)
                ):
                    # If this variable is tracked, must be using an expandable cluster.
                    was_backup = self.test_cluster.was_added_in_backup(  # type: ignore
                        stmt.accessible_object()
                    )
                    if was_backup:
                        needed_expansion = True
                        logger.info("Required test cluster expansion to parse.")
                elif isinstance(stmt, ASTAssignStatement):
                    if stmt.rhs_is_call():
                        needed_calls = True
                    else:
                        needed_uninterp = True

            self._num_added_tests_needed_expansion += 1 if needed_expansion else 0
            self._num_added_tests_needed_uninterp += (
                1 if (needed_calls or needed_uninterp) else 0
            )
            self._num_added_tests_needed_calls += (
                1 if (needed_calls and not needed_uninterp) else 0
            )

            stat.track_output_variable(
                RuntimeVariable.LLMNeededExpansion,
                self._num_added_tests_needed_expansion,
            )

            stat.track_output_variable(
                RuntimeVariable.LLMNeededUninterpreted,
                self._num_added_tests_needed_uninterp,
            )

            stat.track_output_variable(
                RuntimeVariable.LLMNeededUninterpretedCallsOnly,
                self._num_added_tests_needed_calls,
            )

        stat.track_output_variable(
            RuntimeVariable.LLMStageSavedTests, self._num_pytlm_tests_added
        )
        stat.track_output_variable(
            RuntimeVariable.LLMStageSavedMutants, self._num_mutant_pytlm_tests_added
        )

    def generate_tests(self) -> tsc.TestSuiteChromosome:
        self.before_search_start()
        self._goals_manager = _GoalsManager(
            self._test_case_fitness_functions,  # type: ignore
            self._archive,
            self.executor.tracer.get_known_data(),
        )
        self._number_of_goals = len(self._test_case_fitness_functions)
        stat.set_output_variable_for_runtime_variable(
            RuntimeVariable.Goals, self._number_of_goals
        )

        self._population = self._get_random_population()
        # Some code to export the random population
        # exporter = PyTestExporter(wrap_code=False)
        # self._logger.info(
        #     "Initial population test cases:\n %s",
        #     exporter.export_sequences(
        #         "/tmp/initial_seeds.py", [tcc.test_case for tcc in self._population]
        #     ),
        # )
        self._goals_manager.update(self._population)

        # Calculate dominance ranks and crowding distance
        fronts = self._ranking_function.compute_ranking_assignment(
            self._population, self._goals_manager.current_goals
        )
        for i in range(fronts.get_number_of_sub_fronts()):
            fast_epsilon_dominance_assignment(
                fronts.get_sub_front(i), self._goals_manager.current_goals
            )

        self.before_first_search_iteration(
            self.create_test_suite(self._archive.solutions)
        )
        i = 0
        while self.resources_left() and len(self._archive.uncovered_goals) > 0:
            if(i==100):
                self.evolve_targeted(self.create_test_suite(self._archive.solutions))
            # else:
            #     if(i==50):
            #         self.evolve_targeted(self.create_test_suite(self._archive.solutions))
            else:
                self.evolve()
            i+=1
            self.after_search_iteration(self.create_test_suite(self._archive.solutions))

        self.after_search_finish()
        return self.create_test_suite(
            self._archive.solutions
            if len(self._archive.solutions) > 0
            else self._get_best_individuals()
        )

    def evolve(self) -> None:
        """Runs one evolution step."""
        offspring_population: list[
            tcc.TestCaseChromosome
        ] = self._breed_next_generation()

        # Create union of parents and offspring
        union: list[tcc.TestCaseChromosome] = []
        union.extend(self._population)
        union.extend(offspring_population)

        # Ranking the union
        self._logger.debug("Union Size = %d", len(union))
        # Ranking the union using the best rank algorithm
        fronts = self._ranking_function.compute_ranking_assignment(
            union, self._goals_manager.current_goals
        )

        # Form the next population using “preference sorting and non-dominated
        # sorting” on the updated set of goals
        remain = max(
            config.configuration.search_algorithm.population,
            len(fronts.get_sub_front(0)),
        )
        index = 0
        self._population.clear()

        # Obtain the first front
        front = fronts.get_sub_front(index)

        while remain > 0 and remain >= len(front) != 0:
            # Assign crowding distance to individuals
            fast_epsilon_dominance_assignment(front, self._goals_manager.current_goals)
            # Add the individuals of this front
            self._population.extend(front)
            # Decrement remain
            remain -= len(front)
            # Obtain the next front
            index += 1
            if remain > 0:
                front = fronts.get_sub_front(index)

        # Remain is less than len(front[index]), insert only the best one
        if remain > 0 and len(front) != 0:
            fast_epsilon_dominance_assignment(front, self._goals_manager.current_goals)
            front.sort(key=lambda t: t.distance, reverse=True)
            for k in range(remain):
                self._population.append(front[k])

        self._goals_manager.update(self._population)

    def evolve_targeted(self, test_suite: tsc.TestSuiteChromosome):
        """Runs an evolution step that targets uncovered functions.   运行针对未覆盖功能的进化步骤

        Args:
            test_suite: the test suite to base coverage off of.   # 参数， 测试套件以基础覆盖为基础
        """

        original_population: Set[tc.TestCase] = {
            chrom.test_case for chrom in self._population
        }  # 创建一个集合，包含当前种群中的所有测试用例



        #
        if config.configuration.pytlm.target_low_coverage_functions:  # 这个参数为true，就说明要以低覆盖率功能为目标
            print(languagemodelseeding.executor)
            print(languagemodelseeding.model)
            print(languagemodelseeding.test_cluster)
            test_cases = languagemodelseeding.target_uncovered_functions(  # 函数，为test_suit覆盖较少的函数生成测试用例
                test_suite,
                int((config.configuration.pytlm.num_seeds_to_inject)/2),  # inject的种子默认为10
                self.resources_left,
            )
        else:  # 另一种情况，上面的参数不为true，不以低覆盖功能为目标，采用随机的策略生成测试用例，直到资源耗尽
            test_cases = []  # testcases初始为空
            for _ in range(config.configuration.pytlm.num_seeds_to_inject):
                if not self.resources_left():
                    break
                test_cases.extend(languagemodelseeding.get_random_targeted_testcase())

        test_case_chromosomes = [
            tcc.TestCaseChromosome(test_case, self.test_factory)
            for test_case in test_cases
        ]  # 将生成的测试用例转化为TestCaseChromosome，存储在test_case_chromosomes列表中
        new_offspring: List[tcc.TestCaseChromosome] = []   # 创建一个空列表， new_offspring 来存储新的后代
        while (
            len(new_offspring) < config.configuration.search_algorithm.population
            and self.resources_left()
        ):  # 循环，直到达到配置中的种群大小或者资源耗尽
            # print("test_case_chromosomes:    ----------  -----  ")
            # print(test_case_chromosomes)

            # 随机选择两个测试用例染色体，分别克隆它们，得到子代1 和子代2
            offspring_1 = randomness.choice(test_case_chromosomes).clone()

            offspring_2 = randomness.choice(test_case_chromosomes).clone()

            if (
                randomness.next_float()
                <= config.configuration.search_algorithm.crossover_rate  # 依概率 对子代1和子代2进行交叉操作
            ):
                try:
                    self._crossover_function.cross_over(offspring_1, offspring_2)
                except ConstructionFailedException:
                    self._logger.debug("CrossOver failed.")
                    continue

            self._mutate(offspring_1)  # 突变子代1
            if offspring_1.has_changed() and offspring_1.size() > 0:
                new_offspring.append(offspring_1)  # 将子代1加入到新的子代种群中
            self._mutate(offspring_2)  # 突变子代2
            if offspring_2.has_changed() and offspring_2.size() > 0:
                new_offspring.append(offspring_2)  # 将子代2加入到新的子代种群中

        self.evolve_common(test_case_chromosomes + new_offspring)   # 将初始染色体和新生成的后代作为参数进行进一步进化操作

        # 检查中种群中的测试用例是否有新的测试用例，如果有新的测试用例，标记 " added_tests " 为true，并注册这些新增的测试用例
        added_tests = False
        for chrom in self._population:
            test_case = chrom.test_case
            if test_case not in original_population:
                added_tests = True
                # test_cases is the original gene   rated test cases
                mutated = test_case not in test_cases
                self._register_added_testcase(test_case, mutated)
        self._log_num_pytlm_tests_added()


        if not added_tests:
            # If we were unsuccessful in adding tests, double the plateau
            # length so we don't waste too much time querying codex.
            # 如果没有成功添加新的测试用例，则将平台期长度加倍，避免后续迭代中浪费太多时间
            self._plateau_len = 2 * self._plateau_len

    def evolve_common(self, offspring_population) -> None:
        """The core logic to save offspring if they are interesting.

        Args:
            offspring_population: the offspring to try and save
        """

        # Create union of parents and offspring
        union: list[tcc.TestCaseChromosome] = []
        union.extend(self._population)
        union.extend(offspring_population)

        uncovered_goals: OrderedSet[
            ff.FitnessFunction
        ] = self._archive.uncovered_goals  # type: ignore

        # Ranking the union
        self._logger.debug("Union Size = %d", len(union))
        # Ranking the union using the best rank algorithm
        fronts = self._ranking_function.compute_ranking_assignment(
            union, uncovered_goals
        )

        remain = len(self._population)
        index = 0
        self._population.clear()

        # Obtain the next front
        front = fronts.get_sub_front(index)

        while remain > 0 and remain >= len(front) != 0:
            # Assign crowding distance to individuals
            fast_epsilon_dominance_assignment(front, uncovered_goals)
            # Add the individuals of this front
            self._population.extend(front)
            # Decrement remain
            remain -= len(front)
            # Obtain the next front
            index += 1
            if remain > 0:
                front = fronts.get_sub_front(index)

        # Remain is less than len(front[index]), insert only the best one
        if remain > 0 and len(front) != 0:
            fast_epsilon_dominance_assignment(front, uncovered_goals)
            front.sort(key=lambda t: t.distance, reverse=True)
            for k in range(remain):
                self._population.append(front[k])

        self._archive.update(self._population)

class _GoalsManager:
    """Manages goals and provides dynamically selected ones for the generation."""

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        fitness_functions: OrderedSet[ff.FitnessFunction],
        archive: CoverageArchive,
        known_data: KnownData,
    ) -> None:
        self._archive = archive
        branch_fitness_functions: OrderedSet[
            bg.BranchCoverageTestFitness
        ] = OrderedSet()
        for fit in fitness_functions:
            assert isinstance(fit, bg.BranchCoverageTestFitness)
            branch_fitness_functions.add(fit)
        self._graph = _BranchFitnessGraph(branch_fitness_functions, known_data)
        self._current_goals: OrderedSet[
            bg.BranchCoverageTestFitness
        ] = self._graph.root_branches
        self._archive.add_goals(self._current_goals)  # type: ignore

    @property
    def current_goals(self) -> OrderedSet[ff.FitnessFunction]:
        """Provides the set of current goals.

        Returns:
            The set of current goals
        """
        return self._current_goals  # type: ignore

    def update(self, solutions: list[tcc.TestCaseChromosome]) -> None:
        """Updates the information on the current goals from the found solutions.

        Args:
            solutions: The previously found solutions
        """
        # We must keep iterating, as long as new goals are added.
        new_goals_added = True
        while new_goals_added:
            self._archive.update(solutions)
            covered = self._archive.covered_goals
            new_goals: OrderedSet[bg.BranchCoverageTestFitness] = OrderedSet()
            new_goals_added = False
            for old_goal in self._current_goals:
                if old_goal in covered:
                    children = self._graph.get_structural_children(old_goal)
                    for child in children:
                        if child not in self._current_goals and child not in covered:
                            new_goals.add(child)
                            new_goals_added = True
                else:
                    new_goals.add(old_goal)
            self._current_goals = new_goals
            self._archive.add_goals(self._current_goals)  # type: ignore
        self._logger.debug("current goals after update: %s", self._current_goals)


class _BranchFitnessGraph:
    """Best effort re-implementation of EvoSuite's BranchFitnessGraph.

    Arranges the fitness functions for all branches according to their control
    dependencies in the CDG. Each node represents a fitness function. A directed edge
    (u -> v) states that fitness function v should be added for consideration
    only when fitness function u has been covered."""

    def __init__(
        self,
        fitness_functions: OrderedSet[bg.BranchCoverageTestFitness],
        known_data: KnownData,
    ):
        self._graph = nx.DiGraph()
        # Branch less code objects and branches that are not control dependent on other
        # branches.
        self._root_branches: OrderedSet[bg.BranchCoverageTestFitness] = OrderedSet()
        self._build_graph(fitness_functions, known_data)

    def _build_graph(
        self,
        fitness_functions: OrderedSet[bg.BranchCoverageTestFitness],
        known_data: KnownData,
    ):
        """Construct the actual graph from the given fitness functions."""
        for fitness in fitness_functions:
            self._graph.add_node(fitness)

        for fitness in fitness_functions:
            if fitness.goal.is_branchless_code_object:
                self._root_branches.add(fitness)
                continue
            assert fitness.goal.is_branch
            branch_goal = cast(bg.BranchGoal, fitness.goal)
            predicate_meta_data = known_data.existing_predicates[
                branch_goal.predicate_id
            ]
            code_object_meta_data = known_data.existing_code_objects[
                predicate_meta_data.code_object_id
            ]
            if code_object_meta_data.cdg.is_control_dependent_on_root(
                predicate_meta_data.node
            ):
                self._root_branches.add(fitness)

            dependencies = code_object_meta_data.cdg.get_control_dependencies(
                predicate_meta_data.node
            )
            for dependency in dependencies:
                goal = bg.BranchGoal(
                    predicate_meta_data.code_object_id,
                    dependency.predicate_id,
                    dependency.branch_value,
                )
                dependent_ff = self._goal_to_fitness_function(fitness_functions, goal)
                self._graph.add_edge(dependent_ff, fitness)

        # Sanity check
        assert {n for n in self._graph.nodes if self._graph.in_degree(n) == 0}.issubset(
            self._root_branches
        ), "Root branches cannot depend on other branches."

    @property
    def dot(self):
        """Return DOT representation of this graph."""
        dot = to_pydot(self._graph)
        return dot.to_string()

    @property
    def root_branches(self) -> OrderedSet[bg.BranchCoverageTestFitness]:
        """Return the root branches, i.e., the fitness functions that have
        no preconditions."""
        return OrderedSet(self._root_branches)

    @staticmethod
    def _goal_to_fitness_function(
        search_in: OrderedSet[bg.BranchCoverageTestFitness], goal: bg.BranchGoal
    ) -> bg.BranchCoverageTestFitness:
        """Little helper to find the fitness function associated with a certain goal.

        Args:
            search_in: The list to search in
            goal: The goal to search for

        Returns:
            The found fitness function.
        """
        for fitness in search_in:
            if fitness.goal == goal:
                return fitness
        raise RuntimeError(f"Could not find fitness function for goal: {goal}")

    def get_structural_children(
        self, fitness_function: bg.BranchCoverageTestFitness
    ) -> OrderedSet[bg.BranchCoverageTestFitness]:
        """Get the fitness functions that are structural children of the given
        fitness function.

        Args:
            fitness_function: The fitness function whose structural children should be
            returned.

        Returns:
            The structural children fitness functions of the given fitness function.
        """
        return OrderedSet(self._graph.successors(fitness_function))
