import ast


# 1. Parse raw code into a tree
source_code = """
my_list=["apples","oranges"]
print(my_list[1])
"""
tree = ast.parse(source_code)

# 2. View the node structure for debugging
print(ast.dump(tree, indent=4))

# 3. Traverse nodes
for node in ast.walk(tree):
    if isinstance(node, ast.Constant):
        print(f"Found constant value: {node.value}")