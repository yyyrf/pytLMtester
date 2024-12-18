#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Provides a configuration interface for the test generator."""
import dataclasses
import enum
import time

from pynguin.utils.statistics.runtimevariable import RuntimeVariable


class ExportStrategy(str, enum.Enum):
    """Contains all available export strategies.

    These strategies allow to export the generated test cases in different styles,
    such as the style of the `PyTest` framework.  Setting the value to `NONE` will
    prevent exporting of the generated test cases (only reasonable for
    benchmarking, though).
    """

    PY_TEST = "PY_TEST"
    """Export tests in the style of the PyTest framework."""

    NONE = "NONE"
    """Do not export test cases at all."""


class UninterpretedStatementUse(tuple, enum.Enum):
    """Whether to use ASTAssignStatements (aka uninterpreted statements)
    when parsing targeted test cases.
    """

    NONE = (False,)
    """Don't use the statements."""

    ONLY = (True,)
    """Parse test cases with uninterpreted statements, not what the test case
     would have been without uninterpreted statements"""

    BOTH = (True, False)
    """Parse each generated test case with and without uninterpreted statements"""


class Algorithm(str, enum.Enum):
    """Different algorithms supported by Pynguin."""

    DYNAMOSA = "DYNAMOSA"
    """The dynamic many-objective sorting algorithm (cf. Panichella et al. Automated
    test case generation as a many-objective optimisation problem with dynamic selection
    of the targets.  TSE vol. 44 issue 2)."""

    # TODO(ANON): elaborate
    CODAMOSA = "CODAMOSA"
    """MOSA + Codex :)"""



class AssertionGenerator(str, enum.Enum):
    """Different approaches for assertion generation supported by Pynguin."""

    MUTATION_ANALYSIS = "MUTATION_ANALYSIS"
    """Use the mutation analysis approach for assertion generation."""

    SIMPLE = "SIMPLE"
    """Use the simple approach for primitive and none assertion generation."""

    NONE = "NONE"
    """Do not create any assertions."""


class MutationStrategy(str, enum.Enum):
    """Different strategies for creating mutants when using the MUTATION_ANALYSIS
    approach for assertion generation."""

    FIRST_ORDER_MUTANTS = "FIRST_ORDER_MUTANTS"
    """Generate first order mutants."""

    FIRST_TO_LAST = "FIRST_TO_LAST"
    """Higher order mutation strategy FirstToLast.
    (cf. Mateo et al. Validating Second-Order Mutation at System Level. Article.
    IEEE Transactions on SE 39.4 2013)"""

    BETWEEN_OPERATORS = "BETWEEN_OPERATORS"
    """Higher order mutation strategy BetweenOperators.
    (cf. Mateo et al. Validating Second-Order Mutation at System Level. Article.
    IEEE Transactions on SE 39.4 2013)"""

    RANDOM = "RANDOM"
    """Higher order mutation strategy Random.
    (cf. Mateo et al. Validating Second-Order Mutation at System Level. Article.
    IEEE Transactions on SE 39.4 2013)"""

    EACH_CHOICE = "EACH_CHOICE"
    """Higher order mutation strategy EachChoice.
    (cf. Mateo et al. Validating Second-Order Mutation at System Level. Article.
    IEEE Transactions on SE 39.4 2013)"""


class TypeInferenceStrategy(str, enum.Enum):
    """The different available type-inference strategies."""

    NONE = "NONE"
    """Ignore any type information given in the module under test."""

    STUB_FILES = "STUB_FILES"
    """Use type information from stub files."""

    TYPE_HINTS = "TYPE_HINTS"
    """Use type information from type hints in the module under test."""


class StatisticsBackend(str, enum.Enum):
    """The different available statistics backends to write statistics"""

    NONE = "NONE"
    """Do not write any statistics."""

    CONSOLE = "CONSOLE"
    """Write statistics to the standard out."""

    CSV = "CSV"
    """Write statistics to a CSV file."""


class CoverageMetric(str, enum.Enum):
    """The different available coverage metrics available for optimisation"""

    BRANCH = "BRANCH"
    """Calculate how many of the possible branches in the code were executed"""

    LINE = "LINE"
    """Calculate how many of the possible lines in the code were executed"""


class Selection(str, enum.Enum):
    """Different selection algorithms to select from."""

    RANK_SELECTION = "RANK_SELECTION"
    """Rank selection."""

    TOURNAMENT_SELECTION = "TOURNAMENT_SELECTION"
    """Tournament selection.  Use `tournament_size` to set size."""


class TestCaseContext(str, enum.Enum):
    """What kind of extra context to pass to the LLM when
    generating test cases for CodaMOSA"""

    NONE = "NONE"
    """Don't add any additional context."""

    SMALLEST = "SMALLEST"
    """Add the smallest 'winning' test case as context"""

    RANDOM = "RANDOM"
    """Add a random test case as context"""


# pylint:disable=too-many-instance-attributes
@dataclasses.dataclass
class StatisticsOutputConfiguration:
    """Configuration related to output."""

    report_dir: str = "pynguin-report"
    """Directory in which to put HTML and CSV reports"""

    statistics_backend: StatisticsBackend = StatisticsBackend.CSV
    """Which backend to use to collect data"""

    timeline_interval: int = 1 * 1_000_000_000
    """Time interval in nano-seconds for timeline statistics, i.e., we select a data
    point after each interval.  This can be interpolated, if there is no exact
    value stored at the time-step of the interval, see `timeline_interpolation`.
    The default value is every 1.00s."""

    timeline_interpolation: bool = True
    """Interpolate timeline values"""

    coverage_metrics: list[CoverageMetric] = dataclasses.field(
        default_factory=lambda: [
            CoverageMetric.BRANCH,
        ]
    )
    """List of coverage metrics that are optimised during the search"""

    output_variables: list[RuntimeVariable] = dataclasses.field(
        default_factory=lambda: [
            RuntimeVariable.TargetModule,
            RuntimeVariable.Coverage,
        ]
    )
    """List of variables to output to the statistics backend."""

    configuration_id: str = ""
    """Label that identifies the used configuration of Pynguin.  This is only done
    when running experiments."""

    project_name: str = ""
    """Label that identifies the project name of Pynguin.  This is useful when
    running experiments."""

    create_coverage_report: bool = False
    """Create a coverage report for the tested module.
    This can be helpful to find hard to cover parts because Pynguin measures coverage
    on bytecode level which might yield different results when compared with other
    tools, e.g., Coverage.py."""


@dataclasses.dataclass
class TestCaseOutputConfiguration:
    """Configuration related to test case output."""

    output_path: str
    """Path to an output folder for the generated test cases."""

    export_strategy: ExportStrategy = ExportStrategy.PY_TEST
    """The export strategy determines for which test-runner system the
    generated tests should fit."""

    max_length_test_case: int = 2500
    """The maximum number of statement in as test case (normal + assertion
    statements)"""

    assertion_generation: AssertionGenerator = AssertionGenerator.MUTATION_ANALYSIS
    """The generator that shall be used for assertion generation."""

    allow_stale_assertions: bool = False
    """Allow assertion on things that did not change between statement executions."""

    mutation_strategy: MutationStrategy = MutationStrategy.FIRST_ORDER_MUTANTS
    """The strategy that shall be used for creating mutants in the mutation analysis
    assertion generation method."""

    mutation_order: int = 1
    """The order of the generated higher order mutants in the mutation analysis
    assertion generation method."""

    post_process: bool = True
    """Should the results be post processed? For example, truncate test cases after
    statements that raise an exception."""

    float_precision: float = 0.01
    """Precision to use in float comparisons and assertions"""


# pylint:disable=too-many-instance-attributes
@dataclasses.dataclass
class SeedingConfiguration:
    """Configuration related to seeding."""

    seed: int = time.time_ns()
    """A predefined seed value for the random number generator that is used."""

    constant_seeding: bool = True
    """Should the generator use a static constant seeding technique to improve constant
    generation?"""

    large_language_model_seeding: bool = False
    """If set to True, assume we want to use an OpenAI large language
    model to conduct seeding.
    """

    large_language_model_mutation: bool = False
    """If set to True, assume we want to use an OpenAI large language
    model to conduct mutation
    """

    initial_population_seeding: bool = False
    """Should the generator use previously existing testcases to seed the initial
    population?"""

    initial_population_data: str = ""
    """The path to the file with the pre-existing tests. The path has to include the
    file itself."""

    sample_with_replacement: bool = True
    """Should we allow sampling with replacement from previously existing testcases?"""

    allow_expandable_cluster: bool = False
    """Should we create an 'expandable' test cluster, which we can query for new
    functions at seeding/test time?"""

    expand_cluster: bool = False
    """Similar to the above, but create the expanded test cluster from the start.
    """

    remove_testcases_without_coverage: bool = False
    """Should we remove seeded test cases that don't have any coverage of the test
     module?"""

    include_partially_parsable: bool = False
    """If true, keep the parsable parts of seed test cases. If False, only retain test
    cases that are fully parsable. """

    seeded_testcases_reuse_probability: float = 0.9
    """Probability of using seeded testcases when initial population seeding is
    enabled."""

    initial_population_mutations: int = 0
    """Number of how often the testcases collected by initial population seeding should
    be mutated to promote diversity"""

    dynamic_constant_seeding: bool = True
    """Enables seeding of constants at runtime."""

    seeded_primitives_reuse_probability: float = 0.2
    """Probability for using seeded primitive values instead of randomly
    generated ones."""

    seeded_dynamic_values_reuse_probability: float = 0.6
    """Probability of using dynamically seeded values when a primitive seeded
     value will be used."""

    seed_from_archive: bool = False
    """When sampling new test cases reuse some from the archive, if one is used."""

    seed_from_archive_probability: float = 0.2
    """Instead of creating a new test case, reuse a covering solution from the archive,
    iff an archive is used."""

    seed_from_archive_mutations: int = 3
    """Number of mutations applied when sampling from the archive."""

    uninterpreted_statements: UninterpretedStatementUse = UninterpretedStatementUse.NONE
    """Whether to allow uninterpreted assignment statements in the parsed test cases"""


@dataclasses.dataclass
class pytLMConfiguration:
    """Configuration for CodaMosa"""

    authorization_key: str = ""
    """The authorization key to call OpenAI with"""

    model_name: str = ""
    """The OpenAI Model to use for completions"""

    model_base_url: str = ""
    """The base url used to interact with the model.
    Put together, model_base_url and model_relative_url describe
    the url for the model"""

    model_relative_url: str = ""
    """The relative url used to interact with the model.
    Put together, model_base_url and model_relative_url describe
    the url for the model"""

    max_plateau_len: int = 25
    """The number of iterations to let go on before trying to do LLM Seeding"""

    temperature: float = 1
    """The temperature to use when querying the model"""

    num_seeds_to_inject: int = 10
    """Number of seeds to query the OpenAI model for"""

    test_case_context: TestCaseContext = TestCaseContext.NONE
    """What extra context to pass to the LLM when querying for a new test case"""

    target_low_coverage_functions: bool = True
    """Whether or not to target low coverage functions. If false, target random
    functions. 是否针对低覆盖率功能，如果false，则以随机函数为目标"""

    replay_generation_from_file: str = ""
    """Rather than generating new model """



@dataclasses.dataclass
class TypeInferenceConfiguration:
    """Configuration related to type inference."""

    guess_unknown_types: bool = True
    """Should we guess unknown types while constructing parameters?
    This might happen in the following cases:
    The parameter type is unknown, e.g. a parameter is missing a type hint.
    The parameter is not primitive and cannot be created from the test cluster,
    e.g. Callable[...]"""

    type_inference_strategy: TypeInferenceStrategy = TypeInferenceStrategy.TYPE_HINTS
    """The strategy for type-inference that shall be used"""

    max_cluster_recursion: int = 10
    """The maximum level of recursion when calculating the dependencies in the test
    cluster."""

    stub_dir: str = ""
    """Path to the pyi-stub files for the StubInferenceStrategy"""


@dataclasses.dataclass
class TestCreationConfiguration:
    """Configuration related to test creation."""

    max_recursion: int = 10
    """Recursion depth when trying to create objects in a test case."""

    max_delta: int = 20
    """Maximum size of delta for numbers during mutation"""

    max_int: int = 2048
    """Maximum size of randomly generated integers (minimum range = -1 * max)"""

    string_length: int = 20
    """Maximum length of randomly generated strings"""

    bytes_length: int = 20
    """Maximum length of randomly generated bytes"""

    collection_size: int = 5
    """Maximum length of randomly generated collections"""

    primitive_reuse_probability: float = 0.5
    """Probability to reuse an existing primitive in a test case, if available.
    Expects values in [0,1]"""

    object_reuse_probability: float = 0.9
    """Probability to reuse an existing object in a test case, if available.
    Expects values in [0,1]"""

    none_probability: float = 0.1
    """Probability to use None in a test case instead of constructing an object.
    Expects values in [0,1]"""

    skip_optional_parameter_probability: float = 0.7
    """Probability to skip an optional parameter, i.e., do not fill this parameter."""

    max_attempts: int = 1000
    """Number of attempts when generating an object before giving up"""

    insertion_uut: float = 0.5
    """Score for selection of insertion of UUT calls"""

    max_size: int = 100
    """Maximum number of test cases in a test suite"""


@dataclasses.dataclass
class SearchAlgorithmConfiguration:
    """General configuration for search algorithms."""

    min_initial_tests: int = 1
    """Minimum number of tests in initial test suites"""

    max_initial_tests: int = 10
    """Maximum number of tests in initial test suites"""

    population: int = 50
    """Population size of genetic algorithm"""

    chromosome_length: int = 40
    """Maximum length of chromosomes during search"""

    chop_max_length: bool = True
    """Chop statements after exception if length has reached maximum"""

    elite: int = 1
    """Elite size for search algorithm"""

    crossover_rate: float = 0.75
    """Probability of crossover"""

    test_insertion_probability: float = 0.1
    """Initial probability of inserting a new test in a test suite"""

    test_delete_probability: float = 1.0 / 3.0
    """Probability of deleting statements during mutation"""

    test_change_probability: float = 1.0 / 3.0
    """Probability of changing statements during mutation"""

    test_insert_probability: float = 1.0 / 3.0
    """Probability of inserting new statements during mutation"""

    statement_insertion_probability: float = 0.5
    """Initial probability of inserting a new statement in a test case"""

    random_perturbation: float = 0.2
    """Probability to replace a primitive with a random new value rather than adding
    a delta."""

    change_parameter_probability: float = 0.1
    """Probability of replacing parameters when mutating a method or constructor
    statement in a test case.  Expects values in [0,1]"""

    tournament_size: int = 5
    """Number of individuals for tournament selection."""

    rank_bias: float = 1.7
    """Bias for better individuals in rank selection"""

    selection: Selection = Selection.TOURNAMENT_SELECTION
    """The selection operator for genetic algorithms."""

    use_archive: bool = False
    """Some algorithms can be enhanced with an optional archive, e.g. Whole Suite ->
    Whole Suite + Archive. Use this option to enable the usage of an archive.
    Algorithms that always use an archive are not affected by this option."""

    filter_covered_targets_from_test_cluster: bool = False
    """Focus search by filtering out elements from the test cluster when
     they are fully covered."""


@dataclasses.dataclass
class StoppingConfiguration:
    """Configuration related to when Pynguin should stop.
    Note that these are mostly soft-limits rather than hard limits, because
    the search algorithms only check the condition at the start of each algorithm
    iteration."""

    maximum_search_time: int = -1
    """Time (in seconds) that can be used for generating tests."""

    maximum_test_executions: int = -1
    """Maximum number of test cases to be executed."""

    maximum_statement_executions: int = -1
    """Maximum number of test cases to be executed."""

    maximum_iterations: int = -1
    """Maximum iterations"""

    stop_immediately: bool = False
    """Stop immediately after seeding"""


# pylint: disable=too-many-instance-attributes, pointless-string-statement
@dataclasses.dataclass
class Configuration:
    """General configuration for the test generator."""

    project_path: str
    """Path to the project the generator shall create tests for."""

    module_name: str
    """Name of the module for which the generator shall create tests."""

    test_case_output: TestCaseOutputConfiguration
    """Configuration for how test cases should be output."""

    algorithm: Algorithm = Algorithm.DYNAMOSA
    """The algorithm that shall be used for generation."""

    statistics_output: StatisticsOutputConfiguration = dataclasses.field(
        default_factory=StatisticsOutputConfiguration
    )
    """Statistic Output configuration."""

    stopping: StoppingConfiguration = dataclasses.field(
        default_factory=StoppingConfiguration
    )
    """Stopping configuration."""

    seeding: SeedingConfiguration = dataclasses.field(
        default_factory=SeedingConfiguration
    )
    """Seeding configuration."""

    type_inference: TypeInferenceConfiguration = dataclasses.field(
        default_factory=TypeInferenceConfiguration
    )
    """Type inference configuration."""

    test_creation: TestCreationConfiguration = dataclasses.field(
        default_factory=TestCreationConfiguration
    )
    """Test creation configuration."""

    search_algorithm: SearchAlgorithmConfiguration = dataclasses.field(
        default_factory=SearchAlgorithmConfiguration
    )
    """Search algorithm configuration."""

    pytlm: pytLMConfiguration = dataclasses.field(
        default_factory=pytLMConfiguration
    )
    """Condiguration used for CodaMOSA algorithm."""


# Singleton instance of the configuration.
configuration = Configuration(
    project_path="",
    module_name="",
    test_case_output=TestCaseOutputConfiguration(output_path=""),
)
