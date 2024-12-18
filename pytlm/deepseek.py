import re

import openai



openai.api_key = ""
openai.base_url = "https://api.deepseek.com"

def extract_python_code(input_string):
    # 使用正则表达式匹配行 ```python 和行 ``` 之间的内容
    pattern = r'(?<=```python\n).*?(?=\n```)'  # 更新正则表达式
    match = re.search(pattern, input_string, re.DOTALL)

    # 如果找到匹配项，则返回匹配的字符串，否则返回空字符串
    if match:
        return match.group(0)
    else:
        raise RuntimeError(f"Unable to extract Python code from response {input_string}")

def code_deepseek(source_code):
    completion = openai.chat.completions.create(
      model="deepseek-coder",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": source_code + "\\n" + "Please help me indicate the parameter type and return type for each function and class in this code, and output the modified code to me"}

      ],
    )
    explanation_response_str = "\n".join([str(item) for item in completion])

    content = completion.choices[0].message.content

    print(completion.choices[0].message.content)

    return extract_python_code(content)
