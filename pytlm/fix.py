import os
import re
import subprocess
import pynguin.configuration as config
import openai
import pynguin.configuration as config

openai.api_key = ""
openai.base_url = "https://api.deepseek.com"

# 修复已经被标记为错误的测试用例
def fix_xfail():
    a = 0

# 用来获取在执行过程中错误的测试用例
def get_ori_result(output_text):
    # input_string = "FAILED ..\\..\\tmp\\pynguin-results\\test__timer.py::test_case_1 - AssertionError"



    # 正则表达式模式
    pattern = r"FAILED\s+(\S+)\s+-\s+(.*)"

    # 匹配正则表达式
    matches = re.findall(pattern, output_text)

    # 保存匹配结果的列表
    result_list = []

    for match in matches:
        file_path = match[0]
        test_case = file_path.split("::")[-1]
        error_message = match[1]
        result_list.append((file_path, test_case, error_message))

    return result_list

# 获取python文件的导入包的部分
def extract_import_statements(content):
    import_statements = []

    # 使用正则表达式匹配以import或from开头的导入语句
    pattern = r"^(import .*|from .* import .*)$"
    matches = re.findall(pattern, content, re.MULTILINE)

    for match in matches:
        import_statements.append(match)

    return import_statements

# 处理执行失败的测试用例，传入的是测试套件的位置，以及失败的case编号, 返回一个失败测试套件的内容
def generator_fail_function_file(deal_file, fail_function): #
    with open(deal_file, 'r') as f:
        content = f.read()
    # print(content)

    fail_function_str = [f"test_case_{num}" for num in fail_function]

    print(fail_function_str)

    # 提取包含在 fail_function 中的函数
    pattern = r"import .*?|(def (test_case_\d+)\((.|\n)*?)(?=def test_case_|$)"
    matches = re.findall(pattern, content, re.DOTALL)

    function_part = ""

    for match in matches:
        if match[1] in fail_function_str or match[0].startswith("import"):
            code = match[0]
            # 删除 @pytest mark xfail(strict=True) 部分
            code = code.replace("@pytest.mark.xfail(strict=True)", "")
            function_part += code + "\n"

    # print(function_part)

    pack_part = ""

    import_statements = extract_import_statements(content)
    for statement in import_statements:
        pack_part += statement + "\n"

    # print(pack_part)

    new_content = pack_part +"\n"+ function_part

    return new_content


# 使用大模型对测试套件进行修复
def LLM_fix(new_content):
    completion = openai.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": new_content + "\\n" + "This is an incorrect test suite and the reason for its error. Please keep the original function name unchanged while following the logic of the original test case, help me fix it to the correct test case, and output the repaired complete test case"}

        ],
    )
    explanation_response_str = "\n".join([str(item) for item in completion])

    content = completion.choices[0].message.content

    print(completion.choices[0].message.content)

    return content

# 删除测试套件中的注释内容（注释内容大多是无效的代码信息，可以删除）
def remove_comment_lines(content):
    lines = content.split('\n')
    result_lines = []
    for line in lines:
        if not line.strip().startswith('#'):
            result_lines.append(line)

    result_content = '\n'.join(result_lines)
    return result_content

# 检查否存在语法错误  ， 如果存在语法错误，就返回true， 如果没有语法错误就返回false
def check_syntax(content):
    try:
        compile(content, '', 'exec')
        return False  # 没有语法错误
    except SyntaxError:
        return True  # 有语法错误

# 使用大模型进行语法修复
def fix_syntax_error_LLM(content):
    completion = openai.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": content + "\\n" + "There is a syntax error in this code, please help me fix it and output the complete code that has been fixed"}

        ],
    )
    explanation_response_str = "\n".join([str(item) for item in completion])

    content = completion.choices[0].message.content

    print(completion.choices[0].message.content)

    return content

#用来取出 ```Python和```之间的内容
def extract_python_code(content):
    start_tag = "```python"
    end_tag = "```"
    start_index = content.find(start_tag)
    if start_index == -1:
        return "No Python code found."

    end_index = content.find(end_tag, start_index + len(start_tag))
    if end_index == -1:
        return "Missing closing tag '```'."

    return content[start_index + len(start_tag):end_index]



def LLM_fix_runtimeerror(content, module_name, project_path):
    completion = openai.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": content + "\\n" + "The module_name is" + module_name + "\\n" + "The path where the module is located is" + project_path + "\\n" + "Please help me fix this runtime error based on the information I provided above and output the complete code that has been fixed"}

        ],
    )
    explanation_response_str = "\n".join([str(item) for item in completion])

    content = completion.choices[0].message.content

    print(completion.choices[0].message.content)

    return content

def LLM_fix_runtimeerror2(content, module_name, project_path):
    completion = openai.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": "You didn't solve my problem just now, please reconsider \\n" + content + "\\n" + "The module_name is" + module_name + "\\n" + "The path where the module is located is" + project_path + "\\n" + "Please help me fix this runtime error based on the information I provided above and output the complete code that has been fixed"}

        ],
    )
    explanation_response_str = "\n".join([str(item) for item in completion])

    content = completion.choices[0].message.content

    print(completion.choices[0].message.content)

    return content

def LLM_fix_expected_behavior(content):
    completion = openai.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": content + "\\n" }

        ],
    )
    explanation_response_str = "\n".join([str(item) for item in completion])

    content = completion.choices[0].message.content

    print(completion.choices[0].message.content)

    return content


# 第一种修复方案，使用LLM先进行修复
# 如果LLM修复的结果仍然是断言错误，可以考虑删除错误部分的断言
def fix_normal(file_path_tmp, file_name_tmp, fix_filepath_tmp, project_path_tmp, module_name_tmp):
    #file_path = config.configuration.output_path
    file_path = file_path_tmp # "E:\\pynguin-main\\tmp\\pynguin-results\\httpie\\"
    file_name = file_name_tmp # "test_base.py"


    # 修复语法错误
    with open(file_path + file_name, 'r') as f:
        content = f.read()

    # 经过处理之后要进行语法修复的内容
    content = remove_comment_lines(content)

    i = 0
    fixed_with_syntax = False

    while check_syntax(content) and i<10 :
        print("存在语法错误")
        # 调用大模型进行修复
        content = fix_syntax_error_LLM(content)
        content = extract_python_code(content)
        print("第" + str(i) + "次修复：")
        fixed_with_syntax = True
        print(content)
        i+=1
    else:
        print("没有语法错误")

    # print("content: ")
    # print(content)

    if(fixed_with_syntax):
        with open(file_path + file_name, 'w') as f:
            f.write(content)


    result = subprocess.run(['pytest', file_path + file_name], capture_output=True, text=True)
    output_text = result.stdout  # 这个是执行这个测试套件的所有错误信息

    error_text = result.stderr  # 获取运行时错误信息

    # 定义一些常见的运行时错误关键字
    runtime_error_keywords = ["ModuleNotFoundError", "ImportError", "SyntaxError", "IndentationError", "NameError"]

    # 检查运行时错误
    runtime_errors = False

    i=0
    # 检查 stderr 和 stdout 中是否包含运行时错误关键字
    continue_run = True
    while i<5 and continue_run:
        i+=1
        print("修复轮次" + str(i))
        for keyword in runtime_error_keywords:
            if keyword in error_text or keyword in output_text:
                runtime_errors = True
                print(f"执行过程中遇到运行时错误: {keyword}")
                print(output_text)

                project_path = project_path_tmp  # "C:\\Users\\yyrrff\\AppData\\Roaming\\Python\\Python310\\site-packages\\test_n\\httpie\\plugins"
                module_name = module_name_tmp  # "base"

                if(i==1):
                    fix_content = LLM_fix_runtimeerror(content + "\n" + output_text, module_name, project_path)  # 可以考虑将精简的信息传递给大模型，也可以考虑将复杂的信息传递给大模型
                else:
                    fix_content = LLM_fix_runtimeerror2(content + "\n" + output_text, module_name, project_path)

                fix_content = extract_python_code(fix_content)
                continue_run = False
                # print("fix_content ")
                # print(fix_content)

                fix_file = "fix_test_" + extract_variable_content(file_name_tmp, "test_", ".py") + ".py"
                fix_filepath = fix_filepath_tmp  # "E:\\pynguin-main\\tmp\\pynguin-results-fixed\\"
                if not os.path.exists(fix_filepath):
                    os.makedirs(fix_filepath)

                with open(os.path.join(fix_filepath, fix_file), 'w') as f:
                    f.write(fix_content)

                result = subprocess.run(['pytest', fix_filepath + fix_file], capture_output=True, text=True)
                output_text = result.stdout  # 这个是执行这个测试套件的所有错误信息

                error_text = result.stderr  # 获取运行时错误信息

                for keyword in runtime_error_keywords:
                    if keyword in error_text or keyword in output_text:
                        continue_run = True



                # else:
                #     # 情况二  无运行时错误
                #     print("  ")

                print(file_path + file_name)
                # 对运行时错误进行修复,以及预期行为修复

                # failtest_result_list = get_ori_result(output_text)   # 出错的函数列表
                break


    # 如果没有运行时错误，检查测试用例是否通过
    if not runtime_errors:
        if result.returncode == 0:
            print("所有测试用例都成功执行。")
        else:
            print("有测试用例执行失败。")
            print(output_text)  # 输出执行的相关信息，方便调试

    """
    project_path = project_path_tmp # "C:\\Users\\yyrrff\\AppData\\Roaming\\Python\\Python310\\site-packages\\test_n\\httpie\\plugins"
    module_name = module_name_tmp # "base"

    fix_content = LLM_fix_runtimeerror(content + "\n" + output_text, module_name,
                                       project_path)  # 可以考虑将精简的信息传递给大模型，也可以考虑将复杂的信息传递给大模型
    fix_content = extract_python_code(fix_content)
    print("fix_content ")
    print(fix_content)

    fix_file = "fix_test_" + extract_variable_content(file_name_tmp, "test_", ".py") + ".py"
    fix_filepath = fix_filepath_tmp # "E:\\pynguin-main\\tmp\\pynguin-results-fixed\\"
    if not os.path.exists(fix_filepath):
        os.makedirs(fix_filepath)

    with open(os.path.join(fix_filepath, fix_file), 'w') as f:
        f.write(fix_content)
    # else:
    #     # 情况二  无运行时错误
    #     print("  ")


    print(file_path + file_name)
    # 对运行时错误进行修复,以及预期行为修复

    # failtest_result_list = get_ori_result(output_text)   # 出错的函数列表
    """


    '''
    module_name = config.configuration.module_name
    project_path = config.configuration.project_path

    fix_content = LLM_fix_runtimeerror(content + "\n" + output_text, module_name, project_path)  # 可以考虑将精简的信息传递给大模型，也可以考虑将复杂的信息传递给大模型
    print(fix_content)

    '''

    # fix_content = LLM_fix_expected_behavior(content + "\n" + output_text)

    '''
    # 修复运行时错误
    failtest_result_list = get_ori_result(output_text)  # 得到了失败的测试用例的信息
    # 找到具体是哪一个测试用例失败了,保存到失败函数列表中
    # print(failtest_result_list)

    failtest_result = ""  # 这个是这次测试失败的一个精简的信息
    for f in failtest_result_list:
        failtest_result += str(f) + "\n"
    fail_function = []
    for f in failtest_result_list:
        print(f)
        test_case_number = re.search(r'test_case_(\d+)', str(f))

        if test_case_number:
            test_case_number = test_case_number.group(1)
            fail_function.append(test_case_number)
    print(fail_function)
    new_content = generator_fail_function_file(file_path+file_name, fail_function) # 提取出来测试套件中失败的部分，使用大模型进行修复
    # LLM_fix(new_content)
    print(new_content)
    print(failtest_result)
    fix_content = LLM_fix(new_content + "\n" + output_text)  # 可以考虑将精简的信息传递给大模型，也可以考虑将复杂的信息传递给大模型

    fix_file = "fix_" + file_name
    fix_filepath = "D:\\CNNtest\\pynguin-main\\tmp\\pynguin-results-fixed\\"
    with open(fix_filepath + fix_file, 'w') as f:
        f.write(fix_content)
    '''







def extract_variable_content(input_str, prefix, suffix):
    start_idx = input_str.find(prefix) + len(prefix)
    end_idx = input_str.find(suffix, start_idx)
    if start_idx != -1 and end_idx != -1:
        return input_str[start_idx:end_idx]
    else:
        return "Variable content not found in the input string"




if __name__ == '__main__':


    file_path_tmp="E:\\pynguin-main\\tmp\\pynguin-results\\httpie\\"
    # "test_config.py","test_context.py","test_downloads.py",
    file_name_tmps=["test_models.py","test_sessions.py","test_status.py","test_uploads.py"]
    # file_name_tmp="test_manager.py"
    fix_filepath_tmp="E:\\pynguin-main\\tmp\\pynguin-results-fixed\\httpie"
    project_path_tmp = "C:\\Users\\yyrrff\\AppData\\Roaming\\Python\\Python310\\site-packages\\test_n\\httpie\\plugins"
    module_name_tmp = "manager"

    for i, file_name_tmp in enumerate(file_name_tmps):
        module_name_tmp = extract_variable_content(file_name_tmp, "test_", ".py")
        fix_normal(file_path_tmp, file_name_tmp, fix_filepath_tmp, project_path_tmp, module_name_tmp)
        # ss = extract_variable_content(file_name_tmp, "test_", ".py")
        # print(ss)
