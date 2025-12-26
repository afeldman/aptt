#!/usr/bin/env python3
"""Test all internal deepsuite imports without external dependencies."""

import ast
from pathlib import Path
import sys
from typing import List, Set, Tuple


def extract_imports(file_path: Path) -> List[str]:
    """Extract all deepsuite imports from a Python file."""
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("deepsuite"):
                imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("deepsuite"):
                    imports.append(alias.name)
    return imports


def module_to_file(module_name: str, src_dir: Path) -> Path:
    """Convert module name to file path."""
    parts = module_name.split(".")
    # Remove 'deepsuite' prefix
    if parts[0] == "deepsuite":
        parts = parts[1:]

    # Try as package (__init__.py)
    package_path = src_dir / "deepsuite" / "/".join(parts) / "__init__.py"
    if package_path.exists():
        return package_path

    # Try as module (.py)
    module_path = src_dir / "deepsuite" / "/".join(parts[:-1]) / f"{parts[-1]}.py"
    if module_path.exists():
        return module_path

    # Try without last part (parent package)
    if len(parts) > 1:
        parent_path = src_dir / "deepsuite" / "/".join(parts[:-1]) / "__init__.py"
        if parent_path.exists():
            return parent_path

    return None


def main() -> None:
    """Test all imports."""
    src_dir = Path("src")
    if not src_dir.exists():
        print("❌ src/ directory not found")
        sys.exit(1)

    all_py_files = list(src_dir.rglob("*.py"))
    print(f"📁 Found {len(all_py_files)} Python files\n")

    all_imports: Set[str] = set()
    import_errors: list[tuple[str, str]] = []

    # Extract all imports
    for py_file in all_py_files:
        imports = extract_imports(py_file)
        all_imports.update(imports)

    print(f"🔍 Found {len(all_imports)} unique deepsuite imports\n")

    # Validate each import
    print("=" * 60)
    print("VALIDATING IMPORTS")
    print("=" * 60)

    for module_name in sorted(all_imports):
        target_file = module_to_file(module_name, src_dir)
        if target_file is None:
            print(f"❌ {module_name} -> FILE NOT FOUND")
            import_errors.append((module_name, "File not found"))
        else:
            print(f"✅ {module_name}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Valid imports: {len(all_imports) - len(import_errors)}")
    print(f"❌ Invalid imports: {len(import_errors)}")

    if import_errors:
        print("\n🚨 INVALID IMPORTS:")
        for err in import_errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n🎉 All internal imports are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
