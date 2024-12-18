#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2022 Pynguin Contributors
#
#  SPDX-License-Identifier: LGPL-3.0-or-later
#
"""Pynguin is an automated unit test generation framework for Python.

This module provides the main entry location for the program execution from the command
line.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import simple_parsing
from rich.logging import RichHandler
from rich.traceback import install

import pynguin.configuration as config
from pynguin import __version__, deepseek
from pynguin.generator import run_pynguin, set_configuration
from pynguin.utils.console import console


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.UNDERSCORE_AND_DASH,
        description="Pynguin is an automatic unit test generation framework for Python",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbosity",
        default=0,
        help="verbose output (repeat for increased verbosity)",
    )
    parser.add_argument(
        "--log-file",
        "--log_file",
        dest="log_file",
        type=str,
        default=None,
        help="Path to store the log file.",
    )
    parser.add_arguments(config.Configuration, dest="config")

    return parser


def _expand_arguments_if_necessary(arguments: list[str]) -> list[str]:
    """Expand command-line arguments, if necessary.

    This is a hacky way to pass comma separated output variables.  The reason to have
    this is an issue with the automatically-generated bash scripts for Pynguin cluster
    execution, for which I am not able to solve the (I assume) globbing issues.  This
    function allows to provide the output variables either separated by spaces or by
    commas, which works as a work-around for the aforementioned problem.

    This function replaces the commas for the ``--output-variables`` parameter and
    the ``--coverage-metrics`` by spaces that can then be handled by the argument-
    parsing code.

    Args:
        arguments: The list of command-line arguments
    Returns:
        The (potentially) processed list of command-line arguments
    """
    if (
        "--output_variables" not in arguments
        and "--output-variables" not in arguments
        and "--coverage_metrics" not in arguments
        and "--coverage-metrics" not in arguments
    ):
        return arguments
    if "--output_variables" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--output_variables")
    elif "--output-variables" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--output-variables")

    if "--coverage_metrics" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--coverage_metrics")
    elif "--coverage-metrics" in arguments:
        arguments = _parse_comma_separated_option(arguments, "--coverage-metrics")
    return arguments


def _parse_comma_separated_option(arguments: list[str], option: str):
    index = arguments.index(option)
    if "," not in arguments[index + 1]:
        return arguments
    variables = arguments[index + 1].split(",")
    output = arguments[: index + 1] + variables + arguments[index + 2 :]
    return output


def _setup_output_path(output_path: str) -> None:
    path = Path(output_path).resolve()
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _setup_logging(
    verbosity: int,
    log_file: str | None = None,
):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG

    handlers: list[logging.Handler] = []
    if log_file:
        log_file_path = Path(log_file).resolve()
        if not log_file_path.parent.exists():
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    console_handler = RichHandler(
        rich_tracebacks=True, log_time_format="[%X]", console=console
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(console_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s]"
        "(%(name)s:%(funcName)s:%(lineno)d): %(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


# People may wipe their disk, so we give them a heads-up.
_DANGER_ENV = "PYNGUIN_DANGER_AWARE"

def handle_module(module_path: str):
    with open(module_path+".py", "rb") as f:
        content = f.read()

    print("content内容: ")
    print(str(content))

    new_content = deepseek.code_deepseek(str(content))

    print("new_content内容: ")
    print(new_content)

    with open(module_path+"_d.py", 'w') as file:
        file.write(new_content)


def main(argv: list[str] = None) -> int:
    """Entry point for the CLI of the Pynguin automatic unit test generation framework.

    This method behaves like a standard UNIX command-line application, i.e.,
    the return value `0` signals a successful execution.  Any other return value
    signals some errors.  This is, e.g., the case if the framework was not able
    to generate one successfully running test case for the class under test.

    Args:
        argv: List of command-line arguments

    Returns:
        An integer representing the success of the program run.  0 means
        success, all non-zero exit codes indicate errors.
    """
    if _DANGER_ENV not in os.environ:
        print(
            f"""Environment variable '{_DANGER_ENV}' not set.
Aborting to avoid harming your system.
Please refer to the documentation
(https://pynguin.readthedocs.io/en/latest/user/quickstart.html)
to see why this happens and what you must do to prevent it."""
        )
        return -1

    install()
    if argv is None:
        argv = sys.argv
    if len(argv) <= 1:
        argv.append("--help")
    argv = _expand_arguments_if_necessary(argv[1:])

    argument_parser = _create_argument_parser()
    parsed = argument_parser.parse_args(argv)
    # pylint: disable=no-member
    _setup_output_path(parsed.config.test_case_output.output_path)
    # pylint: disable=no-member
    _setup_logging(parsed.verbosity, parsed.log_file)

    # 处理模块
    # handle_module(parsed.config.project_path + "\\" + parsed.config.module_name)
    #
    # parsed.config.module_name = parsed.config.module_name + "_d"

    set_configuration(parsed.config)
    with console.status("Running Pynguin..."):
        return run_pynguin().value


if __name__ == "__main__":
    sys.exit(main(sys.argv))
