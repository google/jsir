r"""Consolidated test generator for JSIR tests.

Scans a directory for test cases (directories containing `input.js`)
and generates test outputs, READMEs, LIT files, and BUILD files based on
`test_config.json` found in the scanned directory.

Usage:
  # Option 1: Direct Python
  python3 maldoca/js/ir/generate_tests.py <directory_to_scan>

  # Option 2: Blaze build and execute binary
  bazel build //maldoca/js/ir:generate_tests
  ./bazel-bin/maldoca/js/ir/generate_tests <directory_to_scan>

Example:
  python3 maldoca/js/ir/generate_tests.py maldoca/js/ir
  # or
  bazel build //maldoca/js/ir:generate_tests
  ./bazel-bin/maldoca/js/ir/generate_tests maldoca/js/ir
"""

import argparse
from concurrent import futures
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional
from tqdm import tqdm


def run_command(cmd: str) -> bytes:
  """Run a shell command and return its output."""
  try:
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.PIPE)
  except subprocess.CalledProcessError as e:
    tqdm.write(
        "Error running command, output:"
        f" {e.output.decode('utf-8', errors='replace')}"
    )
    sys.exit(1)


@dataclasses.dataclass
class TestConfig:
  passes: str
  output_file: str
  prefix: Optional[str] = None
  jsir_analysis: Optional[str] = None
  output_type: Optional[str] = None
  requires_taint_rules: bool = False
  lit_file: str = "run.generated.lit"
  readme_file: str = "README.generated.md"

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "TestConfig":
    return cls(
        passes=data["passes"],
        output_file=data["output_file"],
        prefix=data.get("prefix"),
        jsir_analysis=data.get("jsir_analysis"),
        output_type=data.get("output_type"),
        requires_taint_rules=data.get("requires_taint_rules", False),
        lit_file=data.get("lit_file", "run.generated.lit"),
        readme_file=data.get("readme_file", "README.generated.md"),
    )


@dataclasses.dataclass
class SuiteConfig:
  tests: List[TestConfig]
  build_template: Optional[str] = None
  lit_template: Optional[str] = None
  lit_dest: str = "run.lit"
  generate_readme: bool = True
  overrides: Dict[str, "SuiteConfig"] = dataclasses.field(default_factory=dict)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "SuiteConfig":
    tests = [TestConfig.from_dict(t) for t in data.get("tests", [])]
    overrides = {}
    if "overrides" in data:
      for k, v in data["overrides"].items():
        overrides[k] = SuiteConfig.from_dict(v)

    return cls(
        tests=tests,
        build_template=data.get("build_template"),
        lit_template=data.get("lit_template"),
        lit_dest=data.get("lit_dest", "run.lit"),
        generate_readme=data.get("generate_readme", True),
        overrides=overrides,
    )


def get_extra_args(
    test_config: TestConfig, test_path: str, for_template: bool
) -> str:
  args = []
  if test_config.jsir_analysis:
    args.append(f"--jsir_analysis={test_config.jsir_analysis}")
  if test_config.output_type:
    args.append(f"--output_type={test_config.output_type}")

  if test_config.requires_taint_rules:
    rules_path = os.path.join(test_path, "input_rules.txtpb")
    if for_template:
      args.append(f"--taint_rules_path=$(pwd)/{rules_path}")
    else:
      args.append(f"--taint_rules_path={os.path.abspath(rules_path)}")

  return " " + " ".join(args) if args else ""


def get_check_command(test_config: TestConfig) -> str:
  output_file = test_config.output_file
  prefix = test_config.prefix or "JSIR"
  if test_config.output_type == "analysis":
    return f'diff "$(dirname %s)"/{output_file} -'
  else:
    return f'FileCheck --check-prefix {prefix} "$(dirname %s)"/{output_file}'


def validate_test_config(test_config: TestConfig, test_path: str):
  if test_config.requires_taint_rules:
    rules_path = os.path.join(test_path, "input_rules.txtpb")
    if not os.path.exists(rules_path):
      tqdm.write(
          "Warning: Taint rules file (input_rules.txtpb) not found in"
          f" {test_path}."
      )


def generate_test_file(
    jsir_gen_bin: str,
    input_path: str,
    output_path: str,
    test_config: TestConfig,
):
  passes = test_config.passes
  output_file = test_config.output_file
  prefix = test_config.prefix

  extra_args = get_extra_args(
      test_config, os.path.dirname(input_path), for_template=False
  )

  command = (
      f"{jsir_gen_bin}"
      f" --input_file={os.path.abspath(input_path)}"
      f" --passes='{passes}'"
      f"{extra_args}"
  )

  output = run_command(command).decode("utf-8")
  output = output.strip()

  if prefix:
    lines = output.split("\n")
    out_lines = []
    for i, line in enumerate(lines):
      if not line:
        line = f"// {prefix}-EMPTY:"
      elif i == 0:
        line = f"// {prefix}:      " + line
      else:
        line = f"// {prefix}-NEXT: " + line
      out_lines.append(line)
    output = "\n".join(out_lines) + "\n"
  else:
    output = output + "\n"
    if test_config.output_type == "analysis":
      output += "\n"

  with open(os.path.join(output_path, output_file), "w") as f:
    f.write(output)


def generate_readme_files(test_path: str, tests: List[TestConfig]):
  readme_groups = {}
  for test in tests:
    readme_file = test.readme_file
    if readme_file not in readme_groups:
      readme_groups[readme_file] = []
    readme_groups[readme_file].append(test)

  for readme_file, group_tests in readme_groups.items():
    content = ""
    for test in group_tests:
      lines = [
          "To run manually:",
          "",
          "```shell",
          "bazel run //maldoca/js/ir:jsir_gen -- \\",
          f"  --input_file $(pwd)/{test_path}/input.js \\",
      ]

      extra_args_list = []
      if test.jsir_analysis:
        extra_args_list.append(f"--jsir_analysis {test.jsir_analysis}")
      if test.output_type:
        extra_args_list.append(f"--output_type {test.output_type}")
      if test.requires_taint_rules:
        extra_args_list.append(
            f"--taint_rules_path $(pwd)/{test_path}/input_rules.txtpb"
        )

      if extra_args_list:
        lines.append(f'  --passes "{test.passes}" \\')
        for i, arg in enumerate(extra_args_list):
          if i < len(extra_args_list) - 1:
            lines.append(f"  {arg} \\")
          else:
            lines.append(f"  {arg}")
      else:
        lines.append(f'  --passes "{test.passes}"')

      lines.extend(["```", ""])
      content += "\n".join(lines) + "\n"

    with open(os.path.join(test_path, readme_file), "w") as f:
      f.write(content.strip() + "\n")


def generate_lit_files(test_path: str, tests: List[TestConfig]):
  lit_groups = {}
  for test in tests:
    lit_file = test.lit_file
    if lit_file not in lit_groups:
      lit_groups[lit_file] = []
    lit_groups[lit_file].append(test)

  for lit_file, group_tests in lit_groups.items():
    content = ""
    for test in group_tests:
      lines = [
          "// RUN: CURRENT_FILE_BASENAME=$(basename %s .lit) && \\",
          '// RUN: jsir_gen --input_file "$(dirname %s)"/input.js \\',
          f'// RUN:     --passes "{test.passes}" \\',
      ]

      extra_args_list = []
      if test.jsir_analysis:
        extra_args_list.append(f"--jsir_analysis {test.jsir_analysis}")
      if test.output_type:
        extra_args_list.append(f"--output_type {test.output_type}")
      if test.requires_taint_rules:
        extra_args_list.append(
            f"--taint_rules_path $(pwd)/{test_path}/input_rules.txtpb"
        )

      if extra_args_list:
        for arg in extra_args_list:
          lines.append(f"// RUN:     {arg} \\")

      check_command = get_check_command(test)
      lines.append(f"// RUN:   | {check_command}")

      content += "\n".join(lines) + "\n\n"

    with open(os.path.join(test_path, lit_file), "w") as f:
      f.write(content.strip() + "\n")


def process_test_directory(
    jsir_gen_bin: str,
    test_path: str,
    config: SuiteConfig,
    config_dir: str,
):
  test_name = os.path.basename(test_path)
  tqdm.write(f"Generating test: {test_name}")

  rel_path = os.path.relpath(test_path, config_dir)
  tests = config.tests
  if rel_path in config.overrides:
    tests = config.overrides[rel_path].tests

  for test_config in tests:
    validate_test_config(test_config, test_path)
    generate_test_file(
        jsir_gen_bin=jsir_gen_bin,
        input_path=os.path.join(test_path, "input.js"),
        output_path=test_path,
        test_config=test_config,
    )

  # Handle LIT template/generation
  if config.lit_template:
    lit_template_path = os.path.join(config_dir, config.lit_template)
    shutil.copyfile(lit_template_path, os.path.join(test_path, config.lit_dest))
  else:
    generate_lit_files(test_path, tests)

  # Handle README generation
  if config.generate_readme:
    generate_readme_files(test_path, tests)

  # Handle BUILD template
  if config.build_template:
    build_template_path = os.path.join(config_dir, config.build_template)
    shutil.copyfile(build_template_path, os.path.join(test_path, "BUILD"))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("path", help="Directory to scan for tests")
  args = parser.parse_args()

  root_path = args.path

  # Find all config files
  config_paths = []
  if os.path.exists(os.path.join(root_path, "test_config.json")):
    config_paths.append(os.path.join(root_path, "test_config.json"))
  else:
    # Recursive scan
    for dirpath, _, filenames in os.walk(root_path):
      if "test_config.json" in filenames:
        config_paths.append(os.path.join(dirpath, "test_config.json"))

  if not config_paths:
    print(
        f"Error: No test_config.json found in {root_path} or its"
        " subdirectories",
        file=sys.stderr,
    )
    sys.exit(1)

  tqdm.write("Building jsir_gen...")
  run_command("bazel build //maldoca/js/ir:jsir_gen")
  jsir_gen_bin = os.path.abspath("bazel-bin/maldoca/js/ir/jsir_gen")

  # Flatten tasks
  all_tasks = []
  for config_path in config_paths:
    config_dir = os.path.dirname(config_path)
    with open(config_path, "r") as f:
      config = SuiteConfig.from_dict(json.load(f))

    for test_path, dirnames, file_names_in_dir in os.walk(config_dir):
      # Prune subdirectories that have their own config
      if test_path != config_dir and "test_config.json" in file_names_in_dir:
        dirnames[:] = []
        continue

      if "input.js" in file_names_in_dir:
        all_tasks.append((test_path, config, config_dir))

  with futures.ThreadPoolExecutor() as executor:
    futures_list = [
        executor.submit(
            process_test_directory,
            jsir_gen_bin=jsir_gen_bin,
            test_path=test_path,
            config=config,
            config_dir=config_dir,
        )
        for test_path, config, config_dir in all_tasks
    ]
    for future in tqdm(
        futures.as_completed(futures_list), total=len(futures_list)
    ):
      future.result()


if __name__ == "__main__":
  main()
