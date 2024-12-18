import ast
import re
from importlib.resources import contents
from typing import final

from openai import OpenAI

def extract_python_code(input_string):
    # 使用正则表达式匹配行 ```python 和行 ``` 之间的内容
    pattern = r'(?<=```python\n).*?(?=\n```)'  # 更新正则表达式
    match = re.search(pattern, input_string, re.DOTALL)

    # 如果找到匹配项，则返回匹配的字符串，否则返回空字符串
    if match:
        return match.group(0)
    else:
        return None
        # raise RuntimeError(f"Unable to extract Python code from response {input_string}")

class CodeExtractor(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []
        self.class_stack = []

    def visit_ClassDef(self, node):
        # 将当前类推入栈中
        self.class_stack.append(node)
        class_code = ast.unparse(node)
        self.classes.append(class_code)
        self.generic_visit(node)  # 继续访问类内部的节点
        # 从栈中移除当前类
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        # 如果栈不为空，说明函数定义在类中，不需要额外提取
        if not self.class_stack:
            function_code = ast.unparse(node)
            self.functions.append(function_code)
        self.generic_visit(node)  # 继续访问函数内部的节点

def extract_classes_and_functions(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    extractor = CodeExtractor()
    extractor.visit(tree)

    return {
        'classes': extractor.classes,
        'functions': extractor.functions
    }

def extract_classes_and_functions_content(content):

    tree = ast.parse(content)
    extractor = CodeExtractor()
    extractor.visit(tree)

    return {
        'classes': extractor.classes,
        'functions': extractor.functions
    }


class ImportExtractor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(f"import {alias.name}")
        # 继续访问其他节点
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                self.imports.append(f"from {node.module} import {alias.name}")
        else:
            # 处理相对导入，例如 from .module import name
            for alias in node.names:
                self.imports.append(f"from {node.level*'.'} import {alias.name}")
        # 继续访问其他节点
        self.generic_visit(node)

def extract_imports(file_path):
    with open(file_path, "r") as file:
        source = file.read()
    tree = ast.parse(source, filename=file_path)

    extractor = ImportExtractor()
    extractor.visit(tree)

    # 返回导入列表
    return extractor.imports


def count_token(prompt):
    prompt_length_in_tokens = len(prompt) * 0.3

    return prompt_length_in_tokens
    # 检查是否超过最大长度
    # if prompt_length_in_tokens <= 4000:
    #     return True
    # else:
    #     return False

def generate_class_type(content):
    client = OpenAI(api_key="sk-08f1d963b97246a696f5fbb498304df3", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a python coder"},
            {"role": "user", "content": content + "\n Please help me indicate the parameter type and return type for each function and class in this code, and output the modified code to me."},
        ],
        stream=False
    )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def extract_variable_definitions_ast(file_path):
    """使用 AST 提取 Python 文件中的所有类和函数外的变量定义，并转化为实际代码"""
    variable_definitions = []

    try:
        # 读取并解析 Python 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        # 将 Python 代码转换为 AST（抽象语法树）
        tree = ast.parse(file_content)

        # 获取全局作用域的位置
        global_scope = get_global_scope(tree)

        # 遍历 AST，寻找所有的赋值语句
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # 如果赋值语句位于全局作用域（即不在函数或类内）
                if not is_inside_function_or_class(node, global_scope):
                    for target in node.targets:
                        if isinstance(target, ast.Name):  # 确保赋值的是变量
                            value_code = node_value_to_code(node.value)  # 转换为实际代码
                            variable_definitions.append(f"{target.id} = {value_code}")

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

    return variable_definitions


def get_global_scope(tree):
    """返回全局作用域的范围"""
    global_scope = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # 记录函数和类定义的范围
            global_scope.append(node)
    return global_scope


def is_inside_function_or_class(node, global_scope):
    """判断一个赋值语句是否位于函数或类内部"""
    for scope in global_scope:
        if isinstance(scope, (ast.FunctionDef, ast.ClassDef)):
            # 检查赋值语句的行号是否在函数或类定义内
            if scope.lineno <= node.lineno < scope.end_lineno:
                return True
    return False


def node_value_to_code(node):
    """递归地将 AST 节点转换为实际的代码内容"""
    if isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Tuple):
        return f"({', '.join(node_value_to_code(elt) for elt in node.elts)})"
    elif isinstance(node, ast.List):
        return f"[{', '.join(node_value_to_code(elt) for elt in node.elts)}]"
    elif isinstance(node, ast.Subscript):
        value = node_value_to_code(node.value)
        slice = node_value_to_code(node.slice)
        return f"{value}[{slice}]"
    elif isinstance(node, ast.Call):
        func = node_value_to_code(node.func)
        args = ', '.join(node_value_to_code(arg) for arg in node.args)
        return f"{func}({args})"
    elif isinstance(node, ast.Attribute):
        value = node_value_to_code(node.value)
        return f"{value}.{node.attr}"
    elif isinstance(node, ast.BinOp):
        left = node_value_to_code(node.left)
        right = node_value_to_code(node.right)
        return f"{left} {ast.dump(node.op)} {right}"
    elif isinstance(node, ast.UnaryOp):
        operand = node_value_to_code(node.operand)
        return f"{ast.dump(node.op)}{operand}"
    else:
        return str(node)

def remove_code_blocks(python_code: str, code_blocks: str) -> str:
    """
    Remove code blocks (functions or classes) from a Python code string.

    :param python_code: The content of the Python file (string A).
    :param code_blocks: The content of functions or classes to be removed (string B).
    :return: The modified Python code with the specified code blocks removed.
    """
    # 将代码块的每一行用正则表达式编译，以便匹配整个函数或类定义
    for line in code_blocks.splitlines():
        # 匹配函数或类定义的开始
        # 假设代码块的每一行都是函数或类定义的一部分
        pattern = re.escape(line.strip())  # 转义特殊字符
        pattern = r'^' + pattern + r'.*?$'  # 匹配从行首到行尾的模式
        pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)  # 编译正则表达式

        # 删除匹配的代码块
        python_code, _ = re.subn(pattern, '', python_code)

    return python_code


def extract_non_function_class_code(code: str) -> str:
    """
    从 Python 代码中提取非函数和非类的部分，包括导入语句、常量定义等，
    并删除函数和类的定义以及它们的内部内容（包括 docstring）。
    :param code: 输入的 Python 代码（字符串形式）
    :return: 提取后的代码字符串，不包括函数和类的定义及其内部内容
    """

    # 解析代码并生成 AST（抽象语法树）
    tree = ast.parse(code)

    # 存放非函数和非类的代码行
    non_func_class_code = []

    # 遍历 AST 树
    for node in tree.body:
        if isinstance(node, ast.Import):  # 处理 import 语句
            non_func_class_code.append(ast.unparse(node) + "\n")
        elif isinstance(node, ast.ImportFrom):  # 处理 from ... import ... 语句
            non_func_class_code.append(ast.unparse(node) + "\n")
        elif isinstance(node, ast.Assign):  # 处理常量赋值（常量定义）
            # 处理类型注解的赋值，如 _NO_MAP_TYPES: Set[type] = set()
            for target in node.targets:
                if isinstance(target, ast.Name):  # 如果目标是变量名
                    # 如果目标变量有类型注解，则需要保留该行
                    non_func_class_code.append(ast.unparse(node) + "\n")
        elif isinstance(node, ast.Expr):  # 处理其他顶层表达式
            non_func_class_code.append(ast.unparse(node) + "\n")
        # 跳过函数和类定义及其内容
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

    # 返回合并后的非函数类部分代码
    return "".join(non_func_class_code)


# if __name__ == '__main__':
#     file_path = 'D:\\test_data\\4o-mini\\test_n\\pdir\\attr_category.py'
#     # file_path = 'D:\\test_data\\test_n2\\flutes\\structure.py'
#
#     with open(file_path, "r") as file:
#         content = file.read()
#     # for i in extract_remaining_code(content):
#     #     print(i)
#     print(extract_non_function_class_code(content))

if __name__ == "__main__":
    file_path = 'D:\\test_data\\test_n2\\string_utils\\validation.py'
    extracted_items = extract_classes_and_functions(file_path)
    print("Classes:", len(extracted_items['classes']))
    print("Functions:", len(extracted_items['functions']))
    imports = extract_imports(file_path)
    print("Imports:")
    for imp in imports:
        print(imp)

    with open(file_path, 'r', encoding='utf-8') as file:
        c = file.read()

    # for class_name in extracted_items['classes']:
    #     print(class_name)
    #     remove_substring(c, class_name)
    #
    # for function_name in extracted_items['functions']:
    #     print(function_name)
    #     remove_substring(c, function_name)
    #
    # print(c)
    c = extract_non_function_class_code(c)

    imp_content = ""
    for imp in imports:
        imp_content += imp + "\n"

    final_content = imp_content

    for i in extract_variable_definitions_ast(file_path):
        # print(i)
        final_content += i + "\n"

    final_content = c
    print("--------------------------")
    print(final_content)
    print("--------------------------")


    for cla in extracted_items['classes']:
        content = c
        content += cla
        print(content)
        print("under LLM...")
        new_contents = generate_class_type(content)
        print(new_contents)
        class_contents = extract_python_code(new_contents)
        if class_contents:
            class_contents = extract_classes_and_functions_content(class_contents)
        final_content += "\n" + "\n" + class_contents['classes'][0] + "\n"

    for func in extracted_items['functions']:
        content = c
        content += func
        print(content)
        print("under LLM...")
        new_contents = generate_class_type(content)
        print(new_contents)
        func_contents = extract_python_code(new_contents)
        if func_contents:
            func_contents = extract_classes_and_functions_content(func_contents)
        final_content += "\n" + "\n" + func_contents['functions'][0] + "\n"


    file_path = "D:\\test_data\\test_n2\\string_utils\\validation.py"

    with open(file_path, "w") as file:
        file.write(final_content)

    file_path = 'D:\\test_data\\test_n2\\string_utils\\validation.py'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        print(count_token(content))


