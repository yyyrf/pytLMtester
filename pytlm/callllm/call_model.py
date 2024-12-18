import ast
import inspect
import itertools
import json
import logging
import os
import re
import string
import time
from collections import defaultdict
from typing import Dict, Iterable, List
from typing import Any, Dict, List, Optional, Set
import requests

import pynguin.configuration as config
import pynguin.utils.statistics.statistics as stat
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

logger = logging.getLogger(__name__)



def extract_python_code(input_string):
    # 使用正则表达式匹配行 ```python 和行 ``` 之间的内容
    pattern = r'(?<=```python\n).*?(?=\n```)'  # 更新正则表达式
    match = re.search(pattern, input_string, re.DOTALL)

    # 如果找到匹配项，则返回匹配的字符串，否则返回空字符串
    if match:
        return match.group(0)
    else:
        raise RuntimeError(f"Unable to extract Python code from response {input_string}")

def approx_number_tokens(line: str):

    def char_type(c):
        if c in string.ascii_letters:
            return "letter"
        elif c in string.digits:
            return "digit"
        elif c in string.punctuation:
            return "punctuation"
        elif c in string.whitespace:
            return "whitespace"
        else:
            return "other"

    toks = []
    last_type = "other"
    cur_tok = ""
    for c in line:
        if char_type(c) != last_type:
            toks.append(cur_tok)
            last_type = char_type(c)
            cur_tok = c
        else:
            cur_tok += c
    if len(cur_tok) > 0:
        toks.append(cur_tok)
    return len(toks)

def _openai_api_legacy_request(self, function_header, context, id, difficult_content):
    # TODO: remove this function as part of Issue #19
    url = f"{self._model_base_url}/chat/completions"
    # payload = {
    #     "model": "deepseek-coder",
    #     "id": id,
    #     "messages": [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": context + "\n" + function_header + "\n" + "Please complete the above test function to generate a set of excellent test cases for the target to be tested. Pay special attention to its difficulties. However, please note not to generate too many test cases. Simplification is important.  The content of the test cases should only be given in the form of assert assertions and should not include any additional content.  Do not exceed 10 test cases. And each test case should also be brief. \n " + difficult_content + "\n "},
    #     ],
    #     "stream": False,
    #     # "prompt": context + "\n" + function_header,
    #     "temperature": self._temperature,
    #     # "stop": ["\n# Unit test for", "\ndef ", "\nclass "],  # 停止词会导致没有正确的输出，暂时先去掉
    # }
    payload = {
        "model": "deepseek-coder",
        "id": id,
        # "prompt": context + "\n" + function_header,
        "messages": [
            {"role": "system", "content": "You are a Python expert. Provide in the form of pytest style assertions. Only generate test cases within the function header I gave you. Only 10 test cases need to be generated. There are the following difficulties when generating test cases\n" + difficult_content},
            {"role": "user",
             "content": context + "\n" + function_header + "\n"},
        ],
        # "max_tokens": 200,
        "temperature": self._temperature,
        # "stop": ["\n# Unit test for", "\ndef ", "\nclass "],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self._authorization_key}",
    }
    return url, payload, headers

def _openai_api_legacy_request_per(self, function_header, context):
    # TODO: remove this function as part of Issue #19
    url = f"{self._model_base_url}/chat/completions"
    payload = {
        "model": "deepseek-coder",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context + "\n" + function_header + "\n" + "Based on the above code content, if you want to generate the required test cases, what difficulties will there be? Please list the difficulties in an organized manner, simplify the language as much as possible, and enclose the content of the difficulties in three quotation(\"\"\") marks. "}
        ],
        "stream": False,
        # "prompt": context + "\n" + function_header,
        "temperature": self._temperature,
        # "stop": ["\n# Unit test for", "\ndef ", "\nclass "],  # 停止词会导致没有正确的输出，暂时先去掉
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self._authorization_key}",
    }
    return url, payload, headers

def extract_triple_quoted_string(input_string):
    """
    从输入字符串中提取三引号之间的内容，并将其保存到变量difficulty中返回。

    :param input_string: 包含三引号内容的字符串
    :return: 三引号之间的内容
    """
    import re

    # 使用正则表达式匹配三引号之间的内容
    pattern = r'"""(.*?)"""'
    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        # 找到匹配项，返回第一个非空组的内容
        for group in match.groups():
            if group:
                return group.strip()
    else:
        return None


class _OpenAILanguageModel:

    def __init__(self):
        self._test_src: str
        self._authorization_key: str
        self._complete_model: str
        self._model_base_url: str
        self._model_relative_url: str
        self._edit_model: str
        self._log_path: str = ""
        # TODO(ANON): make configurable; adding a fudge factor
        self._max_query_len = 4000 - 200
        # TODO(ANON): make configurable
        self._temperature: float
        self._token_len_cache = {}
        self.num_codex_calls: int = 0
        self.time_calling_codex: float = 0

    @property
    def temperature(self) -> float:
        """Provides the temperature being used

        Returns:
            the temperature being used
        """
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float):
        self._temperature = temperature

    @property
    def test_src(self) -> str:
        """Provides the source of the module under test

        Returns:
            The source of the module under test
        """
        return self._test_src

    @test_src.setter
    def test_src(self, test_src: str):
        self._test_src = test_src

    @property
    def authorization_key(self) -> str:
        """Provides the authorization key used to query the model

        Returns:
            The organization id
        """
        return self._authorization_key

    @authorization_key.setter
    def authorization_key(self, authorization_key: str):
        self._authorization_key = authorization_key

    @property
    def complete_model(self) -> str:
        """Provides the name of the model used for completion tasks

        Returns:
            The name of the model used for completion tasks
        """
        return self._complete_model

    @complete_model.setter
    def complete_model(self, complete_model: str):
        self._complete_model = complete_model

    @property
    def edit_model(self) -> str:
        """Provides the name of the model used for editing tasks

        Returns:
            The name of the model used for editing tasks
        """
        return self._edit_model

    @edit_model.setter
    def edit_model(self, edit_model: str):
        self._edit_model = edit_model

    @property
    def model_base_url(self) -> str:
        """The base url used to interact with the model. Put together, model_base_url and model_relative_url describe
        the url for the model

        Returns:
            The base url used to interact with the model
        """
        return self._model_base_url

    @model_base_url.setter
    def model_base_url(self, model_base_url: str):
        self._model_base_url = model_base_url

    @property
    def model_relative_url(self) -> str:
        """The relative url used to interact with the model. Put together, model_base_url and model_relative_url
        describe the url for the model

        Returns:
            The relative url used to interact with the model
        """
        return self._model_relative_url

    @model_relative_url.setter
    def model_relative_url(self, model_relative_url: str):
        self._model_relative_url = model_relative_url

    def _get_maximal_source_context(
        self, start_line: int = -1, end_line: int = -1, used_tokens: int = 0
    ):
        """Tries to get the maximal source context that includes start_line to end_line but
        remains under the threshold.

        Args:
            start_line: the start line that should be included
            end_line: the end line that should be included
            used_tokens: the number of tokens to reduce the max allowed by

        Returns:
            as many lines from the source as possible that fit in max_context.
        """

        split_src = self._test_src.split("\n")
        num_lines = len(split_src)

        if end_line == -1:
            end_line = num_lines

        # Return everything if you can
        if (
            sum([self._get_num_tokens_at_line(i) for i in range(1, num_lines + 1)])
            < self._max_query_len
        ):
            return self._test_src

        if (
            sum([self._get_num_tokens_at_line(i) for i in range(1, end_line + 1)])
            < self._max_query_len
        ):
            return "\n".join(split_src[0:end_line])

        # Otherwise greedily take the lines preceding the end line
        cumul_len_of_prefix: List[int] = []
        cumul_len: int = 0
        for i in reversed(range(1, end_line + 1)):
            tok_len = self._get_num_tokens_at_line(i)
            cumul_len += tok_len
            cumul_len_of_prefix.insert(0, cumul_len)

        context_start_line = 0
        for idx, cumul_tok_len in enumerate(cumul_len_of_prefix):
            line_num = idx + 1
            if cumul_tok_len < self._max_query_len - used_tokens:
                context_start_line = line_num
                break

        return "\n".join(split_src[context_start_line:end_line])

    def _call_mutate(self, function_to_mutate: str) -> str:
        """Asks the model to fill in the `??` in the given function

        Args:
            function_to_mutate: a string containing code with a `??` placeholder

        Returns:
            the result of calling the model to edit the given code
        """
        # context = self._get_maximal_source_context(
        #     used_tokens=approx_number_tokens(function_to_mutate)
        # )
        context = ""
        url = f"https://api.openai.com/v1/engines/{self.edit_model}/edits"

        payload = {
            "input": context + "\n" + function_to_mutate,
            "instruction": "Fill in the ??",
            "temperature": self._temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._authorization_key}",
        }
        time_start = time.time()
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        self.time_calling_codex += time.time() - time_start
        self.num_codex_calls += 1
        stat.track_output_variable(RuntimeVariable.LLMCalls, self.num_codex_calls)
        stat.track_output_variable(
            RuntimeVariable.LLMQueryTime, self.time_calling_codex
        )
        if res.status_code != 200:
            logger.error("Failed to call for edit:\n%s", res.json())
            return ""
        return res.json()["choices"][0]["text"]

    def _call_completion(
        self, function_header: str, context_start: int, context_end: int
    ) -> str:
        context = self._get_maximal_source_context(context_start, context_end)
        url0, payload0, headers0 = _openai_api_legacy_request_per(self, function_header, context)
        res_per = requests.post(url0, data=json.dumps(payload0), headers=headers0)
        difficult_content = extract_triple_quoted_string(res_per.json()['choices'][0]['message']['content'])

        print("payload0:----------------------")
        print(json.dumps(payload0))
        print("res_per:-----------------------")
        print(res_per.json())
        print("difficult_content:----------------------")
        print(difficult_content)


        url, payload, headers = _openai_api_legacy_request(
            self, function_header, context, res_per.json()['id'], difficult_content
        )



        time_start = time.time()
        print("url:--------------------------")
        print(url)
        print("payload:--------------------------")
        print(json.dumps(payload))
        print("headers:--------------------------")
        print(headers)

        print("res per:--------------------------")
        print(res_per.json())
        res = requests.post(url, data=json.dumps(payload), headers=headers)
        # print(res.status_code)
        print("res-------------------------------")
        print(res.json())
        print(extract_python_code(res.json()['choices'][0]['message']['content']))
        self.time_calling_codex += time.time() - time_start
        self.num_codex_calls += 1
        stat.track_output_variable(RuntimeVariable.LLMCalls, self.num_codex_calls)
        stat.track_output_variable(
            RuntimeVariable.LLMQueryTime, self.time_calling_codex
        )
        if res.status_code != 200:
            logger.error("Failed to call for completion:\n%s", res.json())
            logger.error(self.complete_model)
            return ""
        return extract_python_code(res.json()['choices'][0]['message']['content'])

    def _get_num_tokens_at_line(self, line_num: int) -> int:
        if len(self._token_len_cache) == 0:
            self._token_len_cache = {
                i + 1: approx_number_tokens(line)
                for i, line in enumerate(self._test_src.split("\n"))
            }
        return self._token_len_cache[line_num]

    def target_test_case(self, gao: GenericCallableAccessibleObject, context="") -> str:

        if gao.is_method():
            method_gao: GenericMethod = gao  # type: ignore
            function_header = (
                f"# Unit test for method {method_gao.method_name} of "
                f"class {method_gao.owner.__name__}\n"  # type: ignore
                f"def test_{method_gao.owner.__name__}"
                f"_{method_gao.method_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(method_gao.owner)  # type: ignore
                end_line = start_line + len(source_lines) - 1
                if (
                    sum(
                        [
                            self._get_num_tokens_at_line(i)
                            for i in range(start_line, end_line + 1)
                        ]
                    )
                    > self._max_query_len
                ):
                    source_lines, start_line = inspect.getsourcelines(method_gao.owner)  # type: ignore
                    end_line = start_line + len(source_lines) - 1
            except (TypeError, OSError):
                start_line, end_line = -1, -1
        elif gao.is_function():
            fn_gao: GenericFunction = gao  # type: ignore
            function_header = (
                f"# Unit test for function {fn_gao.function_name}"
                f"\ndef test_{fn_gao.function_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(fn_gao.callable)
                end_line = start_line + len(source_lines) - 1
            except (TypeError, OSError):
                start_line, end_line = -1, -1
        elif gao.is_constructor():
            constructor_gao: GenericConstructor = gao  # type: ignore
            class_name = constructor_gao.generated_type().__name__  # type: ignore
            function_header = (
                f"# Unit test for constructor of class {class_name}"
                f"\ndef test_{class_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(
                    constructor_gao.generated_type()  # type: ignore
                )
                end_line = start_line + len(source_lines)
            except (TypeError, OSError):
                start_line, end_line = -1, -1

        completion = self._call_completion(
            context + function_header, start_line, end_line
        )
        # Remove any trailing statements that don't parse
        generated_test = fixup_result(function_header + completion)
        generated_tests: Dict[str, str] = rewrite_tests(generated_test)
        for test_name in generated_tests:
            if test_name in function_header:
                return generated_tests[test_name]
        return ""


class FileMockedModel(_OpenAILanguageModel):
    def __init__(self, filename: str):
        assert os.path.isfile(filename)
        self._generation_bank: Dict[str, Iterable[str]] = {}
        self._initialize_contents(filename)
        super().__init__()

    def _initialize_contents(self, filename):
        contents_bank: Dict[str, List[str]] = defaultdict(list)
        with open(filename, encoding="UTF-8") as generations_file:
            all_lines = generations_file.readlines()
            i = 0
            while i < len(all_lines):
                cur_line = all_lines[i]
                if cur_line.startswith("# Generated at "):
                    if i + 2 > len(all_lines):
                        break
                    header = all_lines[i + 1] + all_lines[i + 2].rstrip()
                    i = i + 3
                    contents = []
                    while i < len(all_lines) and not all_lines[i].startswith(
                        "# Generated at "
                    ):
                        contents.append(all_lines[i])
                        i = i + 1
                    contents_bank[header].append("".join(contents))
                else:
                    i = i + 1
        for header, contents_lst in contents_bank.items():
            if len(contents_lst) > 0:
                self._generation_bank[header] = itertools.cycle(contents_lst)

    def _call_completion(
        self, function_header: str, context_start: int, context_end: int
    ) -> str:
        if function_header in self._generation_bank:
            ret_value = "\n" + next(self._generation_bank[function_header])  # type: ignore
            return ret_value
        else:
            return "\npass\n"


languagemodel = _OpenAILanguageModel()


def is_expr_or_stmt(node: ast.AST):
    return isinstance(node, ast.expr) or isinstance(node, ast.stmt)


def has_call(node: ast.AST):

    class CallFinder(ast.NodeVisitor):
        def __init__(self):
            super().__init__()
            self.has_call = False

        def visit_Call(self, call: ast.Call):
            self.has_call = True

    finder = CallFinder()
    finder.visit(node)
    return finder.has_call


def key_in_dict(value, d):
    if isinstance(value, bool):
        return any([k is value for k in d.keys()])
    else:
        return value in d


def has_bound_variables(node: ast.AST, bound_variables: Set[str]) -> bool:

    class BoundVariableVisitor(ast.NodeVisitor):

        def __init__(self):
            self.has_bound_variable = False

        def visit_Name(self, node: ast.Name):
            if node.id in bound_variables:
                self.has_bound_variable = True

    bound_variable_visitor = BoundVariableVisitor()
    bound_variable_visitor.visit(node)
    return bound_variable_visitor.has_bound_variable


class StmtRewriter(ast.NodeTransformer):

    def __init__(self):
        self.stmts_to_add: List[ast.stmt] = []
        self._bound_variables: Set[str] = set()
        self.replace_only_free_subnodes = False

        self._bound_variables_stack: List[Set[str]] = []
        self._replace_only_free_stack: List[bool] = []

        self.used_varnames: Set[str] = set()
        self.var_counter = 0
        self.constant_dict = {}
        self.used_varnames_stack: List[Set[str]] = []
        self.var_counter_stack: List[int] = []
        self.constant_dict_stack: List[Dict[Any, ast.Name]] = []
        super().__init__()


    def reset_stmts_to_add(self):
        self.stmts_to_add = []

    def fresh_varname(self):
        new_varname = "var_" + str(self.var_counter)
        self.var_counter += 1
        while new_varname in self.used_varnames:
            # In case var_X is already defined in this test
            new_varname = "var_" + str(self.var_counter)
            self.var_counter += 1
        self.used_varnames.add(new_varname)
        return new_varname

    def replace_with_varname(self, node):
        if isinstance(node, ast.Name):
            return node
        if isinstance(node, ast.Constant) and key_in_dict(
            node.value, self.constant_dict
        ):
            varname = self.constant_dict[node.value]
        elif self.replace_only_free_subnodes and has_bound_variables(
            node, self._bound_variables
        ):
            return node
        else:
            varname = self.fresh_varname()
            if isinstance(node, ast.Constant):
                self.constant_dict[node.value] = varname
            assign_decl = ast.Assign(
                targets=[ast.Name(varname, ctx=ast.Store())], value=node
            )
            self.stmts_to_add.append(assign_decl)

        name_node = ast.Name(varname, ctx=ast.Load())
        return name_node

    def enter_new_block_scope(self):
        self.used_varnames_stack.append(self.used_varnames)
        self.var_counter_stack.append(self.var_counter)
        self.constant_dict_stack.append(self.constant_dict)
        self.used_varnames = set()
        self.var_counter = 0
        self.constant_dict = {}

    def exit_block_scope(self):
        self.used_varnames = self.used_varnames_stack.pop()
        self.var_counter = self.var_counter_stack.pop()
        self.constant_dict = self.constant_dict_stack.pop()

    def enter_new_bound_scope(self):
        self._bound_variables_stack.append(set(self._bound_variables))
        self._replace_only_free_stack.append(self.replace_only_free_subnodes)
        self.replace_only_free_subnodes = True

    def exit_bound_scope(self):
        self._bound_variables = self._bound_variables_stack.pop()
        self.replace_only_free_subnodes = self._replace_only_free_stack.pop()

    def get_stmts_to_add(self):
        return self.stmts_to_add

    def visit_block_helper(self, block: List[ast.stmt]):
        self.enter_new_block_scope()
        new_body = []
        for stmt in block:
            new_stmt = self.visit(stmt)
            new_body.extend(self.get_stmts_to_add())
            self.reset_stmts_to_add()
            if new_stmt is not None:
                new_body.append(new_stmt)
        self.exit_block_scope()
        return new_body

    def generic_visit(self, node):
        field_assign = {}
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                new_value_lst = []
                for item in value:
                    if is_expr_or_stmt(item):
                        new_item = self.visit(item)
                        item_name = self.replace_with_varname(new_item)
                        new_value_lst.append(item_name)
                    else:
                        new_value_lst.append(item)
                field_assign[field] = new_value_lst
            elif is_expr_or_stmt(value):
                new_value = self.visit(value)
                value_name = self.replace_with_varname(new_value)
                field_assign[field] = value_name
            else:
                field_assign[field] = value
        return node.__class__(**field_assign)

    def visit_only_calls_subnodes(self, node):
        field_assign = {}
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                new_value_lst = []
                for item in value:
                    if is_expr_or_stmt(item) and has_call(item):
                        new_item = self.visit(item)
                        item_name = self.replace_with_varname(new_item)
                        new_value_lst.append(item_name)
                    else:
                        new_value_lst.append(item)
                field_assign[field] = new_value_lst
            elif is_expr_or_stmt(value) and has_call(value):
                new_value = self.visit(value)
                value_name = self.replace_with_varname(new_value)
                field_assign[field] = value_name
            else:
                field_assign[field] = value
        return node.__class__(**field_assign)


    def visit_Call(self, call: ast.Call):
        func = self.visit(call.func)
        if not isinstance(func, ast.Attribute):
            func = self.replace_with_varname(func)
        new_args = []
        for arg in call.args:
            if isinstance(arg, ast.Starred):
                new_args.append(self.visit(arg))
            else:
                arg_value = self.visit(arg)
                new_args.append(self.replace_with_varname(arg_value))
        new_kwargs = []
        for kwarg in call.keywords:
            kwarg_value = self.visit(kwarg.value)
            kwarg_value = self.replace_with_varname(kwarg_value)
            new_kwargs.append(ast.keyword(arg=kwarg.arg, value=kwarg_value))

        return ast.Call(func=func, args=new_args, keywords=new_kwargs)

    def visit_Subscript(self, subscript: ast.Subscript):
        if isinstance(subscript.slice, ast.Tuple):
            new_slice_elts = []
            for elem in subscript.slice.elts:
                new_elem = self.visit(elem)
                if isinstance(elem, ast.Slice):
                    new_slice_elts.append(new_elem)
                else:
                    new_slice_elts.append(self.replace_with_varname(new_elem))
            new_slice = ast.Tuple(elts=new_slice_elts, ctx=ast.Load())
        elif isinstance(subscript.slice, ast.Slice):
            new_slice = self.visit(subscript.slice)
        else:
            new_slice = self.visit(subscript.slice)
            new_slice = self.replace_with_varname(new_slice)

        new_value = self.visit(subscript.value)

        return ast.Subscript(value=new_value, slice=new_slice, ctx=subscript.ctx)

    def visit_UnaryOp(self, node):
        if isinstance(node.operand, ast.Constant):
            return node
        else:
            return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        value_visited = self.visit(node.value)
        if isinstance(node.value, ast.Attribute):
            node.value = value_visited
        else:
            node.value = self.replace_with_varname(value_visited)
        return node

    def visit_Assign(self, assign: ast.Assign):
        for target in assign.targets:
            if isinstance(target, ast.Name):
                self.used_varnames.add(target.id)
        new_rhs = self.visit(assign.value)
        return ast.Assign(
            targets=assign.targets, value=new_rhs, type_comment=assign.type_comment
        )

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None:
            return self.visit(ast.Assign(targets=[node.target], value=node.value))
        else:
            return None

    def visit_AugAssign(self, node):
        new_aug_assign = self.generic_visit(node)
        rhs_binop = ast.BinOp(
            left=new_aug_assign.target, op=new_aug_assign.op, right=new_aug_assign.value
        )
        return ast.Assign(targets=[new_aug_assign.target], value=rhs_binop)

    def visit_NamedExpr(self, node: ast.NamedExpr):
        rhs = self.visit(node.value)
        self.stmts_to_add.append(ast.Assign(targets=[node.target], value=rhs))
        return node.target

    def visit_Expr(self, expr: ast.Expr):
        if isinstance(expr.value, ast.NamedExpr):
            rhs = self.visit(expr.value.value)
            return ast.Assign(targets=[expr.value.target], value=rhs)
        # Don't mess with awaits/yields
        if type(expr.value) in (ast.Await, ast.Yield, ast.YieldFrom):
            return expr
        rhs = self.visit(expr.value)
        return ast.Assign(
            targets=[ast.Name(id=self.fresh_varname(), ctx=ast.Store)], value=rhs
        )

    def visit_Assert(self, assert_node: ast.Assert):
        if isinstance(assert_node.test, ast.Call):
            return self.generic_visit(assert_node)
        else:
            new_test = self.visit_only_calls_subnodes(assert_node.test)
        return ast.Assert(new_test)

    def visit_FunctionDef(self, fn_def_node: ast.FunctionDef):
        if not fn_def_node.name.startswith("test_"):
            return fn_def_node

        # Visit the main body
        new_body = self.visit_block_helper(fn_def_node.body)
        fn_def_node.body = new_body
        ast.fix_missing_locations(fn_def_node)

        return fn_def_node

    def visit_ClassDef(self, node: ast.ClassDef):
        if any(
            [
                isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_")
                for stmt in node.body
            ]
        ):
            new_body = []
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_"):
                    new_body.append(rewrite_test(stmt))
                else:
                    new_body.append(stmt)
            return ast.ClassDef(
                name=node.name,
                bases=node.bases,
                keywords=node.keywords,
                body=new_body,
                decorator_list=node.decorator_list,
            )
        return node

    def visit_For(self, node):
        node.body = self.visit_block_helper(node.body)
        node.orelse = self.visit_block_helper(node.orelse)
        return node

    def visit_While(self, node: ast.While):
        node.body = self.visit_block_helper(node.body)
        node.orelse = self.visit_block_helper(node.orelse)
        return node

    def visit_If(self, node):
        node.body = self.visit_block_helper(node.body)
        node.orelse = self.visit_block_helper(node.orelse)
        return node

    def visit_With(self, node):
        node.body = self.visit_block_helper(node.body)
        return node

    def visit_Try(self, node: ast.Try):
        node.body = self.visit_block_helper(node.body)
        node.orelse = self.visit_block_helper(node.orelse)
        node.finalbody = self.visit_block_helper(node.finalbody)
        return node

    def visit_Lambda(self, node: ast.Lambda):
        self.enter_new_bound_scope()
        all_args: ast.arguments = node.args
        for arg in all_args.args + all_args.kwonlyargs:
            arg_name = arg.arg
            self._bound_variables.add(arg_name)
        if all_args.kwarg is not None:
            self._bound_variables.add(all_args.kwarg.arg)
        if all_args.vararg is not None:
            self._bound_variables.add(all_args.vararg.arg)
        new_lambda = self.generic_visit(node)
        self.exit_bound_scope()
        return new_lambda

    def get_comprehension_bound_vars(self, node: ast.comprehension) -> List[str]:
        return [elem.id for elem in ast.walk(node.target) if isinstance(elem, ast.Name)]

    def _visit_generators_common(self, generators: List[ast.comprehension]):
        new_generators = []
        for comp in generators:
            self._bound_variables.update(self.get_comprehension_bound_vars(comp))
            new_generators.append(self.visit(comp))
        return new_generators

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        self.enter_new_bound_scope()
        new_generators = self._visit_generators_common(node.generators)
        new_elt = self.visit(node.elt)
        ret_val = ast.GeneratorExp(elt=new_elt, generators=new_generators)
        self.exit_bound_scope()
        return ret_val

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        self.enter_new_bound_scope()
        new_generators = self._visit_generators_common(node.generators)
        new_elt = self.visit(node.elt)
        ret_val = ast.ListComp(elt=new_elt, generators=new_generators)
        self.exit_bound_scope()
        return ret_val

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        self.enter_new_bound_scope()
        new_generators = self._visit_generators_common(node.generators)
        new_elt = self.visit(node.elt)
        ret_val = ast.SetComp(elt=new_elt, generators=new_generators)
        self.exit_bound_scope()
        return ret_val

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        self.enter_new_bound_scope()
        new_generators = self._visit_generators_common(node.generators)
        new_key = self.visit(node.key)
        new_value = self.visit(node.value)
        ret_val = ast.DictComp(key=new_key, value=new_value, generators=new_generators)
        self.exit_bound_scope()
        return ret_val

    ## Things we want to leave unmodified ##

    def visit_Import(self, node):
        return node

    def visit_ImportFrom(self, node):
        return node

    def visit_Await(self, node):
        return node

    def visit_AsyncFunctionDef(self, node):
        return node

    def visit_AsyncFor(self, node):
        return node

    def visit_AsyncWith(self, node):
        return node

    def visit_Match(self, node):
        return node


def rewrite_test(fn_def_node: ast.FunctionDef):
    visitor = StmtRewriter()
    visitor.visit(fn_def_node)
    return fn_def_node


def fixup_result(result):
    try:
        ast.parse(result)
        return result
    except SyntaxError as e:
        line_to_rm = e.lineno
        lines = result.split("\n")
        if line_to_rm is None or line_to_rm >= len(lines):
            return fixup_result("\n".join(lines[:-1]))
        else:
            return fixup_result("\n".join(lines[:line_to_rm]))


def rewrite_tests(source: str) -> Dict[str, str]:
    source = fixup_result(source)
    module_node: ast.Module = ast.parse(source)
    assert isinstance(module_node, ast.Module)
    # Rewrite the tests
    return_tests: Dict[str, str] = {}
    for child_node in module_node.body:
        if isinstance(child_node, ast.FunctionDef) and child_node.name.startswith(
            "test_"
        ):
            test_module = ast.Module(
                body=[rewrite_test(child_node)], type_ignores=module_node.type_ignores
            )
            test_module = ast.fix_missing_locations(test_module)
            try:
                return_tests[child_node.name] = ast.unparse(test_module) + "\n"
            except AttributeError as e:
                # Info until we don't need to replicate this
                logger.info("error")

    # print(return_tests)

    return return_tests


def fixup_imports(test_case_str: str, node: Optional[ast.Module] = None):
    if node is None:
        node = ast.parse(test_case_str)
    imports: List[ast.Import] = [
        elem for elem in node.body if isinstance(elem, ast.Import)
    ]
    quals_to_replace = {}
    for import_ in imports:
        for name in import_.names:
            if name.asname is None:
                continue
            if config.configuration.module_name in name.name:
                quals_to_replace[name.asname + "."] = ""
            else:
                pass
                # quals_to_replace[name.asname + "."] = name.name + "."
    test_case_str = "\n".join(
        [
            line
            for line in test_case_str.split("\n")
            if f"import {config.configuration.module_name}" not in line
        ]
    )
    for alias_to_replace, replace_name in quals_to_replace.items():
        test_case_str = test_case_str.replace(alias_to_replace, replace_name)
    return test_case_str

