"""Check that the package is a move-only split of the original script.

Every top-level function, class and assignment of the original must appear exactly once in the package (or in the
entry script), with an identical abstract syntax tree once config.<NAME> is read back as <NAME> for the five run-time
settings and the 'global' statement of main() is ignored. Comments and blank lines do not enter the comparison.
"""
import ast
import sys
from pathlib import Path

MUTABLE = {"DATA_DIR", "WORK_DIR", "OUT_DIR", "N_BOOT", "REFIT"}
orig, files = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]


class Undo(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "config" and node.attr in MUTABLE:
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node

    def visit_Global(self, node):
        return None


def key(n):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        return n.name
    if isinstance(n, (ast.Assign, ast.AnnAssign)):
        return ast.unparse(n.targets[0] if isinstance(n, ast.Assign) else n.target)
    return None


def items(tree):
    out = {}
    for n in tree.body:
        k = key(n)
        if k:
            assert k not in out, f"duplicate {k}"
            out[k] = ast.dump(Undo().visit(n), include_attributes=False)
    return out


old = items(ast.parse(orig.read_text()))
new = {}
for f in files:
    for k, v in items(ast.parse(f.read_text())).items():
        assert k not in new, f"{k} defined twice (second in {f.name})"
        new[k] = (v, f.name)

missing = sorted(set(old) - set(new))
extra = sorted(set(new) - set(old))
diff = sorted(k for k in set(old) & set(new) if old[k] != new[k][0])
print(f"original top-level definitions: {len(old)}; found in package: {len(set(old) & set(new))}")
print("missing:", missing or "none")
print("not in the original:", extra or "none")
print("changed:", diff or "none")
sys.exit(1 if missing or diff else 0)
