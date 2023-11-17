import ast
import os

# NOT FUNCTIONAL 
# Asked AI to create description for each files and each functions in it; but it produces this:

def extract_functions(filename):
    with open(filename, "rt") as file:
        tree = ast.parse(file.read())

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_info = {
                "name": node.name,
                "description": ast.get_docstring(node),
                "dependencies": [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
            }
            functions.append(function_info)
    return functions

def generate_report(directory):
    report = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                report[filepath] = extract_functions(filepath)
    return report

report = generate_report("/")
print(report)