Currently maldoca/astgen is written in C++. Let's rewrite it in python.

Note:

* First make sure that py_{binary,library,test} works in this repo.

* All tests must be ported and still pass. In particular:

  * maldoca/astgen/test contains several golden test files
  
  * maldoca/js contains a lot of generated files

* Please do this piece by piece, in whatever order you think is reasonable.

  Whenever you ported one component, write or port the corresponding tests.

---

## Progress Log

### 2026-07-03 — Investigation + plan

Investigated `maldoca/astgen` (C++) in depth. Summary of what it is:

* `ast_gen_main` is an offline codegen binary (not wired into any Bazel
  `genrule` — run by hand, output checked in). It reads a schema
  (`ast_def.textproto`, parsed via `ast_def.proto`/`type.proto`) describing
  an AST (node classes, fields, enums, union types), and emits:
  - `ast.generated.h` / `ast.generated.cc` — C++ AST node classes
  - `ast_to_json.generated.cc` — `Serialize()` to `nlohmann::json`
  - `ast_from_json.generated.cc` — `FromJson()` parse from `nlohmann::json`
  - (optional, if `--ir_path` given) `<lang>ir_ops.generated.td` (MLIR ODS),
    `conversion/ast_to_<lang>ir.generated.cc`,
    `conversion/<lang>ir_to_ast.generated.cc` (AST<->MLIR IR lowering/raising)
  - `ast_ts_interface.generated` (TS-flavored interface doc; only invoked by
    tests, not by `ast_gen_main`)
* Core model: `Symbol` (word-list identifier, case conversion),
  `Type`/`NonListType`/`ListType`/... (field type system with per-target
  printers: `CcType`, `JsType`, `TdType`, `CcMlirBuilderType`, ...),
  `AstDef`/`NodeDef`/`FieldDef`/`EnumDef` (semantic schema model built from
  the proto, with inheritance graph, topological sort, validation).
* Printers are visitor-style classes over `AstDef` that print one output
  file each: `ast_header_printer`, `ast_source_printer`,
  `ast_serialize_printer`, `ast_from_json_printer`, `ts_interface_printer`,
  `ir_table_gen_printer`, `ast_to_ir_source_printer`, `ir_to_ast_source_printer`.
* Correctness contract for tests: `maldoca/astgen/test/{assign,enum,lambda,
  list,multiple_inheritance,region,typed_lambda,union,variant}/` each has a
  hand-written `ast_def.textproto` input plus checked-in golden output files
  for every printer above. `test/ast_gen_test_util.cc` runs each printer and
  string-diffs (whitespace-stripped) against the goldens — this is the thing
  to replicate in Python, printer by printer.
* `test/*/conversion/conversion_test.cc` additionally compiles the generated
  C++ against real MLIR and round-trips AST->IR->AST — this validates the
  *generated C++ code*, not the generator itself; since we're targeting
  byte-for-byte (whitespace-insensitive) identical output to the existing
  goldens, we don't need to reproduce this test in Python — it stays as a
  C++ test consuming the same golden `.generated.*` files.
* Important finding: `maldoca/js/ast/ast_def.textproto` (the real schema
  that produced the checked-in `maldoca/js/ast/*.generated.*` /
  `maldoca/js/ir/*.generated.*` files) does **not exist** in this repo
  snapshot. So we cannot regenerate the real JS AST from scratch right now;
  the port's correctness will be judged against the 9 small golden test
  cases under `maldoca/astgen/test/`, which is a complete enough contract
  for all printers.
* Python build tooling: `rules_python` + a `pip.parse` hub are already
  declared in `MODULE.bazel` (Python 3.11), but there are currently **zero**
  `py_binary`/`py_library` targets anywhere in the repo, and the only
  `py_test` usages are inside `bazel/lit.bzl` (MLIR `lit` runner glue). So
  step 1 of the task (confirm `py_{binary,library,test}` works) is a real
  gap to close, not just a formality.

Plan (piece by piece, porting tests alongside each piece):

1. [ ] Smoke-test `py_binary`/`py_library`/`py_test` actually work end to end
   in this Bazel setup (trivial target, `bazel test`).
2. [ ] Port `Symbol` -> `symbol.py` + `symbol_test.py`.
3. [ ] Port `Type` system -> `type.py` + `type_test.py`. Reuse
   `type.proto`/`ast_def.proto` as-is via `py_proto_library` (avoids
   rewriting the schema/textproto format — goldens' input files stay valid).
4. [ ] Port `AstDef`/`NodeDef`/`FieldDef`/`EnumDef` -> `ast_def.py` (+ tests
   covering graph/topological-sort/validation logic previously in C++ tests).
5. [ ] Port shared printer infra: `printer_base.py`, `cc_printer_base.py`,
   `ast_gen_utils.py`.
6. [ ] Port printers one at a time, each validated against the 9 golden
   test-case directories (whitespace-insensitive diff, matching
   `ast_gen_test_util.cc`'s approach):
   - [ ] `ast_header_printer.py`
   - [ ] `ast_source_printer.py`
   - [ ] `ast_serialize_printer.py`
   - [ ] `ast_from_json_printer.py`
   - [ ] `ts_interface_printer.py`
   - [ ] `ir_table_gen_printer.py`
   - [ ] `ast_to_ir_source_printer.py`
   - [ ] `ir_to_ast_source_printer.py`
7. [ ] Port `ast_gen_main.cc` -> `ast_gen_main.py` (`py_binary` CLI, same
   flags: `--ast_def_path`, `--cc_namespace`, `--ast_path`, `--ir_path`).
8. [ ] Wire up `py_test` targets per golden test-case dir mirroring the
   existing `cc_test(ast_gen_test)` ones, confirm all pass.
9. [ ] Decide fate of the C++ implementation (keep both during transition,
   or remove C++ once Python fully matches) — revisit with user before
   deleting anything.

Location decision: writing the Python sources alongside the existing C++
sources in `maldoca/astgen/` (e.g. `symbol.py` next to `symbol.cc`), with
`py_library`/`py_test` rules added into the existing `BUILD` files. Keeps
each ported piece co-located with its C++ counterpart during the transition.

### 2026-07-03 — Step 1 done: py_binary/py_library/py_test confirmed working

Built a throwaway `//_smoketest` package (py_library + py_binary + py_test),
ran `bazel test` and `bazel run` successfully, then deleted it. Confirmed
`rules_python` 1.0.0 (Python 3.11) from `MODULE.bazel` works end to end in
this repo even though no real `py_*` targets existed before now.

### 2026-07-03 — Step 2 done: Symbol ported

Added `maldoca/astgen/symbol.py` (port of `symbol.h`/`symbol.cc`) and
`maldoca/astgen/symbol_test.py` (port of `symbol_test.cc`, same 7 cases,
using `unittest`). Added `py_library(symbol_py)` /
`py_test(symbol_py_test)` rules to `maldoca/astgen/BUILD` next to the
existing `cc_library(symbol)` / `cc_test(symbol_test)`.

API is a straight method-for-method port (`ToSnakeCase` ->
`to_snake_case()`, etc.) using Python `snake_case` method naming since this
is idiomatic Python, not a literal identifier-for-identifier port. Same
parsing algorithm, same reserved-keyword list (kept even though it's
C++-flavored, since generated identifiers still need to avoid C++/MLIR
keywords).

Verified: `bazel test //maldoca/astgen:symbol_py_test
//maldoca/astgen:symbol_test` — both pass (7/7 each).

Next: port the `Type` system (`type.proto`/`type.h`/`type.cc`), reusing the
existing `.proto` via `py_proto_library` rather than rewriting the schema
format.

### 2026-07-03 — Step 3 done: Type system ported

Confirmed `py_proto_library` (from
`@com_google_protobuf//bazel:py_proto_library.bzl`) works in this repo:
added `py_proto_library(type_py_pb2)` / `py_proto_library(ast_def_py_pb2)`
to `maldoca/astgen/BUILD`, generating Python bindings straight from the
existing `type.proto`/`ast_def.proto` — no schema rewrite needed, and the
existing `.textproto` golden inputs stay valid as-is. One gotcha found via a
throwaway smoke test: the proto2 oneof field named `class` (in
`TypePb`/`ScalarTypePb`/`NonListTypePb`) is *not* renamed to `class_` in the
Python bindings (unlike the C++ bindings) — since `class` is a hard Python
keyword, it has to be accessed via `getattr(pb, "class")`/`pb.WhichOneof(...)`
rather than attribute syntax. Also confirmed the generated `_pb2` modules
pull in the protobuf Python runtime transitively — no extra pip/requirements
entries needed.

Added `maldoca/astgen/type.py` (port of `type.h`/`type.cc`) and
`maldoca/astgen/type_test.py` (port of `type_test.cc`, all 10 `TEST()`
cases). Design notes:
- The C++ class hierarchy (`Type` -> `NonListType`/`ListType` ->
  `ScalarType`/`VariantType` -> `BuiltinType`/`EnumType`/`ClassType`) is
  kept as-is using Python `abc.ABC` + `abstractmethod`, since printers
  downstream dispatch on concrete subclass behavior the same way the C++
  virtual functions do.
- Dropped the LLVM-style `IsA<T>()` RTTI helper — Python's built-in
  `isinstance()` does the same job natively.
- C++ overloads like `CcType()` / `CcType(MaybeNull)` / `CcType(Optionalness)`
  became a single method `cc_type(optionalness=OPTIONALNESS_UNSPECIFIED)`
  backed by an abstract `_cc_type()` each subclass implements — Python can't
  overload on argument type/count, so the overload set collapses into one
  method with a default argument. Same pattern for `cc_getter_type`/
  `td_type`. Callers needing the `MaybeNull` variant convert via
  `_maybe_null_to_optionalness()` at the call site (matches the two
  remaining real call sites found via grep across all printer `.cc` files).
- `FieldKind`/`Optionalness` are used directly from the generated
  `ast_def_pb2` module (not reimplemented as Python enums) so the values
  stay wire-compatible with the schema.
- `ClassType.node_def` (was `ClassType::node_def_`, C++ `friend class
  AstDef`-only mutable) is just a plain public attribute in Python, set by
  `AstDef` once the schema is resolved (next step) — no need for a friend
  mechanism.

Verified: `bazel test //maldoca/astgen:type_py_test` — 10/10 pass, matching
all 10 `TEST()` cases in `type_test.cc` exactly (same input textprotos, same
expected output strings).

Next: port `AstDef`/`NodeDef`/`FieldDef`/`EnumDef` (`ast_def.h`/`ast_def.cc`)
— the semantic schema model with inheritance graph, topological sort, and
validation. This is the biggest remaining foundational piece; printers
depend on it.

### 2026-07-03 — Step 4 done: AstDef core model ported

Added `maldoca/astgen/ast_def.py` (port of `ast_def.h`/`ast_def.cc`):
`EnumMemberDef`/`EnumDef` (frozen dataclasses), `FieldDef` (dataclass),
`NodeDef` (plain mutable class — mirrors the C++ "only `AstDef` populates
the graph-derived fields" pattern, just without the `friend class`
mechanism since Python has no private/friend access control), and `AstDef`
itself with `from_proto()` doing the same multi-pass construction as
`AstDef::FromProto()`: build nodes -> wire parent pointers -> synthesize
union-type nodes -> compute ancestors/aggregated_fields/children/
descendants/leaves/aggregated_kinds/aggregated_mlir_traits -> topologically
sort by (parents + field-type) dependencies -> synthesize `node_type_enum`
for root types with children -> resolve `ClassType.node_def` back-pointers.

Renamed one thing for a Python-specific reason: C++'s private
`NodeDef::ir_op_name_` (a stored field) and public `NodeDef::ir_op_name()`
(a method) can share a name via C++ overload resolution; Python cannot have
both, so the stored field is `NodeDef.custom_ir_op_name` and the method
stays `ir_op_name()`.

There's no `ast_def_test.cc` in the C++ code — `AstDef::FromProto()` is only
exercised indirectly through the golden printer tests. Since the graph
algorithms (topological sort reused for both ancestors *and* descendants,
with subtly different resulting orders) are intricate and easy to get
wrong, wrote `maldoca/astgen/ast_def_test.py` from scratch with 14 focused
tests: ancestors/descendants/leaves/children ordering, aggregated_fields,
`node_type_enum` synthesis for root types, union-type-as-parent, field-type
topological dependencies, `ClassType.node_def` resolution, `ir_op_name`/
`ir_op_mnemonic` (leaf/non-leaf/custom-override cases), and 5 schema
validation error cases (duplicate node, missing parent, missing union
member, non-camelCase field, non-PascalCase enum).

Two of my first-draft test expectations were wrong and had to be corrected
against the actual (verified-correct) algorithm output rather than
intuition:
- The IR dialect prefix is `lang_name + "ir"` via plain string
  concatenation (`"la" + "ir" = "lair"`), which `Symbol()` parses as *one*
  fused word, not `"la"` + `"ir"` as separate words — confirmed against the
  real checked-in `maldoca/astgen/test/lambda/lair_dialect.td` (dialect name
  is literally "lair").
- `descendants` reuses the exact same DFS-postorder topological-sort helper
  as `ancestors`, just walking `children` edges instead of `parents` edges.
  Because of the "skip if already in sorted_dependencies" short-circuit,
  the resulting order is order-of-first-encounter-dependent, not a clean
  "shallowest first" order — e.g. for `CatDog <: Cat, Dog <: Animal`,
  `Animal.descendants == [CatDog, Cat, Dog]` (CatDog first, since it's
  fully expanded via the Cat branch before the Dog branch reaches it and
  short-circuits). This is inherent to the original C++ algorithm, not a
  porting bug — verified by hand-tracing the same DFS logic.

Verified: `bazel test //maldoca/astgen:ast_def_py_test` — 14/14 pass.

Next: port the shared printer infrastructure (`printer_base.h`,
`cc_printer_base.h`/`.cc`, `ast_gen_utils.h`) — these underpin all 8
printers. The C++ printers are built on `google::protobuf::io::Printer`
(`$var$` substitution + indentation); Python has no equivalent built in, so
this step needs a small from-scratch `Printer` helper before the printers
themselves can be ported.

### 2026-07-03 — Step 5 done: shared printer infrastructure ported

Added `maldoca/astgen/printer_base.py`: a from-scratch `Printer` class
reimplementing the subset of `google::protobuf::io::Printer` the astgen
printers actually use — `$var$` substitution (`$$` -> literal `$`),
2-space-per-level indentation applied at the start of each output line
(matching a specific real detail: blank lines get *no* indentation
whitespace, avoiding spurious trailing whitespace), and `with_indent()`/
`with_vars()` as Python context managers standing in for C++'s
`WithIndent()`/`WithVars()` RAII scopes. No dedicated C++ test exists for
`google::protobuf::io::Printer` in this repo (it's an upstream protobuf
type), so wrote `printer_base_test.py` from scratch (12 tests) pinning the
indentation/substitution/scoping behavior directly, since every downstream
printer's correctness depends on it.

Added `maldoca/astgen/cc_printer_base.py` (port of `cc_printer_base.h`/
`.cc`) and `maldoca/astgen/ast_gen_utils.py` (port of `ast_gen_utils.h`:
`TabPrinter`, `IfStmtPrinter`, `un_indented_source()`,
`field_is_argument()`/`field_is_region()`). `TabPrinter` became a context
manager (`with TabPrinter(options) as tab:`) since Python has no
deterministic destructors to rely on for the postfix-on-scope-exit
behavior the C++ RAII version used.

Wrote `cc_printer_base_test.py` from scratch (7 tests, cross-checked
against literal snippets from `maldoca/astgen/test/lambda/ast.generated.h`)
and caught two real bugs this way before they could propagate into every
downstream printer:
- A typo: `"namespace $cc_namespace_piece${"` was missing the space before
  `{`, which would have produced `namespace maldoca{` instead of
  `namespace maldoca {` in every generated file.
- A genuine discrepancy in the *original C++*: `PrintIncludeHeaders`'s doc
  comment claims it "prints headers in alphabetical order by sorting a copy
  of the header paths," but the actual C++ implementation never calls a
  sort — it just prints the input order. All 4 call sites happen to pass
  already-alphabetized literal lists, which is presumably how the comment
  went stale without anyone noticing. Ported to match the real (unsorted)
  behavior, not the comment, with a note explaining why — sorting here
  would be an unfaithful "fix" that risks silently reordering output.

Verified: `bazel test //maldoca/astgen:printer_base_py_test
//maldoca/astgen:cc_printer_base_py_test` — 12/12 and 7/7 pass.

Next: the 8 printer modules themselves (`ast_header_printer`,
`ast_source_printer`, `ast_serialize_printer`, `ast_from_json_printer`,
`ts_interface_printer`, `ir_table_gen_printer`, `ast_to_ir_source_printer`,
`ir_to_ast_source_printer`), validated against the 9 golden test-case
directories under `maldoca/astgen/test/`.

### 2026-07-03 — Step 6 done: ast_header_printer ported and verified byte-for-byte

Added `maldoca/astgen/ast_header_printer.py` (port of
`ast_header_printer.h`/`.cc`) — prints `ast.generated.h`. Straightforward
method-for-method port onto the now-complete `Printer`/`CcPrinterBase`
infrastructure.

One infra gap found while porting: `Printer.with_indent()` only supported a
fixed 2-space step, but `PrintConstructor()` uses `WithIndent(4)` for
wrapped constructor argument lists. Generalized `printer_base.py` to track
an indent *stack* of per-scope widths (summed for the current line) instead
of a level counter, so `with_indent(4)` works alongside the default
`with_indent()` (2 spaces). Re-ran `printer_base_py_test`/
`cc_printer_base_py_test` after the refactor — still 12/12 and 7/7.

**Golden-file verification strategy**: rather than building out the full
py_test harness (mirroring `ast_gen_test_util.cc`) after each individual
printer, verify each printer against all 9 golden test-case directories via
a throwaway scratch script (`maldoca/astgen/_scratch/`, deleted before the
branch is done), then defer wiring the permanent `py_test` targets to the
final step once all 8 printers exist. This avoids revisiting the same 9
test files 8 separate times.

Needed one Bazel structural fix to make golden files reachable at all:
each of the 9 test-case directories (`maldoca/astgen/test/lambda/`, etc.)
is its own Bazel package (has its own `BUILD` file), so `glob()` from the
parent `maldoca/astgen/test` package cannot see into them — glob never
crosses package boundaries, regardless of visibility. Added a
`filegroup(name = "testdata", srcs = glob(["**"], exclude = ["BUILD"]))` to
each of the 9 subdirectory `BUILD` files (with explicit
`visibility = ["//maldoca/astgen:__subpackages__"]`, since 3 of the 9 don't
set a `default_visibility` at all) so Python test targets anywhere under
`//maldoca/astgen/...` can depend on them.

Verified via the scratch harness: `print_ast_header()` output matches all 9
golden `ast.generated.h` files exactly (whitespace-stripped, same as the
C++ test's comparison). One harness-only bug surfaced and fixed along the
way (not a printer bug): the scratch script initially passed the
runfiles-resolved absolute path as `ast_path` (which feeds into the header
guard), instead of the canonical repo-relative path
`maldoca/astgen/test/<case>` the C++ tests use — caused a
header-guard-only mismatch across all 9 cases, fixed by passing the
canonical path for guard computation while still reading files from the
real resolved directory.

Next: `ast_source_printer` (prints `ast.generated.cc`).

### 2026-07-03 — Step 7 done: ast_source_printer ported and verified byte-for-byte

Added `maldoca/astgen/ast_source_printer.py` (port of
`ast_source_printer.h`/`.cc`) — prints `ast.generated.cc` (getter/setter/
constructor definitions, enum-to-string conversion functions).

Two infra additions needed:
- `Printer.indent()`/`outdent()` (raw, non-context-manager versions)
  alongside `with_indent()`: `PrintConstructor`'s initializer-list
  `TabPrinter` calls `Indent()` in its prefix callback and `Outdent()` in
  its postfix callback, which don't nest as a single lexical block (the
  indent starts partway through the loop, on the first item, and ends when
  the `TabPrinter` itself goes out of scope) — a plain `with_indent()`
  block can't express that, so added the raw pair for this one case.
- A minimal `_cescape()` helper (not a full `absl::CEscape` port — just the
  handful of escapes that actually appear in the 9 golden schemas:
  backslash, quote, and whitespace controls). Verified this was sufficient
  by grepping every `string_value` across all 9 `ast_def.textproto` files
  first, rather than guessing; `test/enum` in particular exercises a
  literal backslash and a tab character, so this wasn't just a
  theoretical case.

One dead-code observation, not reproduced: the C++ `PrintNode()` opens a
`WithVars` scope binding `"NodeType"` to
`ClassType(...).CcType()` (i.e. `"std::unique_ptr<...>"`), but every actual
use of `$NodeType$` downstream is inside a narrower scope that immediately
rebinds it to the correct plain class name — so the outer binding is
provably unobservable dead code. Omitted it rather than reproducing dead
state, unlike the `PrintIncludeHeaders` sorting case (step 5) where the
"bug" *was* observable in the output and had to be matched exactly.

Verified via the scratch harness: `print_ast_source()` output matches all 9
golden `ast.generated.cc` files exactly, including the backslash/tab
escaping edge cases in `test/enum`.

Next: `ast_serialize_printer` (prints `ast_to_json.generated.cc`).

### 2026-07-03 — Step 8 done: ast_serialize_printer ported and verified byte-for-byte

Added `maldoca/astgen/ast_serialize_printer.py` (port of
`ast_serialize_printer.h`/`.cc`) — prints `ast_to_json.generated.cc`
(`Serialize()`/`SerializeFields()` definitions). Matched all 9 golden files
on the first try (no bugs found this round). Noted and skipped one more
dead-code detail while reading the source: `AstSerializePrinter` declares
`PrintSerializeFunctionOverload()` in the header but never defines or calls
it anywhere — an unused, unimplemented method (harmless in C++ since
nothing references it); omitted from the Python port since it has zero
observable effect.

Verified via the scratch harness: `print_ast_to_json()` output matches all
9 golden `ast_to_json.generated.cc` files exactly.

Next: `ast_from_json_printer` (prints `ast_from_json.generated.cc`).

### 2026-07-03 — Step 9 done: ast_from_json_printer ported and verified byte-for-byte

Added `maldoca/astgen/ast_from_json_printer.py` (port of the largest
printer, `ast_from_json_printer.h`/`.cc`) — prints
`ast_from_json.generated.cc` (`FromJson()` factory functions + type
checkers + per-field `GetXxxFromJson()` getters). Matched all 9 golden
files on the first try.

Made `type.maybe_null_to_optionalness()` (previously private, added in
step 3) public — this printer needed the same `MaybeNull` -> `Optionalness`
conversion at its one real external call site
(`PrintListConverter`/`print_list_converter`, converting a list's
`element_maybe_null` before computing the element's `cc_type()`), confirming
the step-3 prediction that this was the only other call site.

Used a `try/finally` in `print_type_checker()` to port `PrintTypeChecker`'s
C++ `absl::Cleanup` (which exists specifically to print the closing `}`
after an early `return` inside the function body) — same pattern as the
`with`-block-can't-express-non-lexical-scope situation from step 7, just
via `try/finally` instead of raw indent/outdent since here it's the
*closing brace* that needs deferring, not an indent level.

Verified via the scratch harness: `print_ast_from_json()` output matches
all 9 golden `ast_from_json.generated.cc` files exactly.

Next: `ts_interface_printer` (prints `ast_ts_interface.generated`).

### 2026-07-03 — Step 10 done: ts_interface_printer ported and verified byte-for-byte

Added `maldoca/astgen/ts_interface_printer.py` (port of
`ts_interface_printer.h`/`.cc`) — prints `ast_ts_interface.generated`, the
human-readable TS-flavored interface doc used only by tests (not by
`ast_gen_main`). Extends `Printer` directly (not `CcPrinterBase`), matching
the C++ class extending `AstGenPrinterBase` directly, since it's not
emitting C++ code. One thing to note for later (mirrored, not fixed): its
`PrintAst` iterates `ast.node_names` (original schema definition order) and
looks up `ast.nodes[name]`, unlike every other printer so far, which
iterates `ast.topological_sorted_nodes` — preserved exactly since it's
observable in output order.

Verified via the scratch harness: `print_ts_interface()` output matches all
9 golden `ast_ts_interface.generated` files exactly.

5 of 8 printers now ported and verified. The remaining 3
(`ir_table_gen_printer`, `ast_to_ir_source_printer`,
`ir_to_ast_source_printer`) are the MLIR-facing ones — expect these to be
the most intricate, since they encode the AST<->IR lowering/raising rules.

Next: `ir_table_gen_printer` (prints `<lang>ir_ops.generated.td`, MLIR ODS).

### 2026-07-03 — Step 11 done: ir_table_gen_printer ported and verified byte-for-byte

Added `maldoca/astgen/ir_table_gen_printer.py` (port of
`ir_table_gen_printer.h`/`.cc`) — prints `<lang>ir_ops.generated.td`, one
MLIR ODS `def <Op> : <Dialect>_Op<...>` per (node, `FieldKind`) pair.
Matched all 6 golden `.generated.td` files (of the 9 test cases, only 6
have an MLIR/IR side — `multiple_inheritance`, `union`, `typed_lambda`
don't) on the first try, despite this being the densest printer yet (trait
computation, variadic-operand counting, region-vs-argument classification).

Confirmed the large `UnIndentedSource(R"(...)")`-wrapped comment/template
blocks in this file (the `*_region_end` explanatory comment and the two
`ExprRegionEndOp`/`ExprsRegionEndOp` op templates) needed real unindenting
this time (unlike the license/`_STRING_TO_ENUM_BODY` blocks in earlier
printers, which were already at zero indent in the C++ source) — hand-simulated
`UnIndentedSource`'s "strip the common leading whitespace" algorithm once on
paper to get the final canonical text, then hardcoded that as a Python
constant (same "no need to call the utility at runtime, since the output is
static" approach as before).

Verified via the scratch harness: `print_ir_table_gen()` output matches all
6 golden `<lang>ir_ops.generated.td` files exactly.

6 of 8 printers done. Next up are the two hardest: `ast_to_ir_source_printer`
and `ir_to_ast_source_printer` (AST<->IR lowering/raising C++ visitor code).

Next: `ast_to_ir_source_printer` (prints `conversion/ast_to_<lang>ir.generated.cc`).

### 2026-07-03 — Step 12 done: ast_to_ir_source_printer ported and verified byte-for-byte

Added `maldoca/astgen/ast_to_ir_source_printer.py` (port of the densest
printer, `ast_to_ir_source_printer.h`/`.cc`) — prints
`conversion/ast_to_<lang>ir.generated.cc`, the C++ visitor code that lowers
AST nodes to MLIR ops/values/attributes. Ported the `Action`/`RefOrVal` C++
enums as Python `enum.Enum`s local to this module (they're printer-specific
concepts, not part of the shared type system). Matched all 6 golden files
on the first try.

One closure-semantics detail worth noting: `PrintRegion`'s C++
`populate_region` lambda captures `rhs` *by reference* (`[&]`), so mutating
`rhs` in the enclosing scope before a later call changes what the lambda
sees — this is exactly how Python closures already behave when a nested
function only *reads* an enclosing-scope variable (no `nonlocal` needed,
since the reassignment happens in the *outer* scope, not inside the nested
function) — so the direct translation just worked without needing any
extra plumbing.

Needed to add a `testdata` filegroup to 6 more `BUILD` files this round: the
`conversion/` subdirectory under each of the 6 IR-enabled test cases (e.g.
`maldoca/astgen/test/lambda/conversion/`) is *also* its own Bazel package
(has its own `BUILD` file, referenced by label from the parent dir's
`cc_test`), so needed the same filegroup-per-package treatment as the
top-level 9 (see step 6).

Verified via the scratch harness: `print_ast_to_ir_source()` output matches
all 6 golden `conversion/ast_to_<lang>ir.generated.cc` files exactly.

Next: `ir_to_ast_source_printer` (prints `conversion/<lang>ir_to_ast.generated.cc`)
— the last printer.

### 2026-07-03 — Step 13 done: ir_to_ast_source_printer ported — all 8 printers complete

Added `maldoca/astgen/ir_to_ast_source_printer.py` (port of
`ir_to_ast_source_printer.h`/`.cc`) — the mirror image of
`ast_to_ir_source_printer`, prints `conversion/<lang>ir_to_ast.generated.cc`
(IR -> AST raising visitor code). Structurally similar to the AST->IR
printer but without the `Action` enum (uses plain recursive `MaybeNull`
dispatch instead).

Discovered while wiring up the verification: unlike the other 5 IR-enabled
test cases, `test/lambda` has no `*ir_to_ast.generated.cc` golden at all —
confirmed by checking `test/lambda/conversion/` directory contents and
cross-referencing `test/lambda/ast_gen_test.cc`, which sets
`expected_ast_to_ir_source_path` but not `expected_ir_to_ast_source_path`.
So this printer is verified against 5 golden files (assign, enum, list,
region, variant), not 6.

Verified via the scratch harness: `print_ir_to_ast_source()` output matches
all 5 golden `conversion/<lang>ir_to_ast.generated.cc` files exactly.

**All 8 printers are now ported and byte-for-byte verified against every
applicable golden file in `maldoca/astgen/test/`.** Every printer matched
on the first real attempt (after the two infra bugs caught in step 5)
— the foundation work (Symbol, Type, AstDef, Printer) paid off by making
each subsequent printer close to mechanical translation.

Next: `ast_gen_main` (the CLI binary entry point), then wire up the
permanent `py_test` golden-diff targets (replacing the `_scratch/` throwaway
harness) as the final step.

### 2026-07-03 — Step 14 done: ast_gen_main CLI ported and end-to-end verified

Added `maldoca/astgen/ast_gen_main.py` (port of `ast_gen_main.cc`) as a
`py_binary` (`//maldoca/astgen:ast_gen_main_py`) — the CLI entry point that
reads an `ast_def.textproto`, runs all the printers, and writes the 4 (or 7,
if `--ir_path` given) output files. Used `argparse` instead of
`absl::Flag`/abseil-py (no abseil-py dependency is pinned in
`requirements.txt`, and there's no other precedent for it in this repo, so
stdlib `argparse` is the simpler/more consistent choice). Made the 3
required flags (`--ast_def_path`, `--cc_namespace`, `--ast_path`) actually
`required=True` in argparse rather than silently defaulting to `""` like
the C++ `ABSL_FLAG` declarations do — a deliberate, low-risk UX
improvement since this binary sits outside the golden-tested output
contract (nothing compares its stdout/stderr against a fixed string).

Verified end-to-end (not just via printer functions in isolation): ran
`bazel run //maldoca/astgen:ast_gen_main_py` against
`maldoca/astgen/test/lambda/ast_def.textproto`, writing to a scratch
directory, then diffed every output file against the real checked-in
goldens. Every difference was exactly and only the header-guard/`#include`
text that's *supposed* to differ (since the scratch run used a different
`--ast_path`/`--ir_path` than what originally produced the goldens) — every
other line matched exactly, confirming the CLI plumbing (arg parsing, file
reading, path joining, file writing for all 7 output files across the
`--ir_path` branch) is correct, not just the underlying printer functions.

Next (final step): wire up permanent `py_test` golden-diff targets per
test-case directory (mirroring each `cc_test(ast_gen_test)`), replacing the
`_scratch/` throwaway verification harness used throughout steps 6-13.

### 2026-07-03 — Step 15 done: permanent py_test golden harness — MIGRATION COMPLETE

Added `maldoca/astgen/test/ast_gen_test_util.py` (port of
`ast_gen_test_util.h`/`.cc`): an `AstGenTest` base class + `AstGenTestParam`
dataclass mirroring the C++ `TestWithParam<AstGenTestParam>` fixture (one
`test_*` method per printer). Added one concrete `ast_gen_test.py` per
test-case directory (9 total: `assign`, `enum`, `lambda`, `list`,
`multiple_inheritance`, `region`, `typed_lambda`, `union`, `variant`), each
setting `PARAM` to the exact values transcribed from that directory's
`ast_gen_test.cc` `INSTANTIATE_TEST_SUITE_P` block. Added a `py_test` rule
to each of the 9 `BUILD` files, `data`-depending on that directory's
`testdata` filegroup (and, for the 6 IR-enabled cases, the `conversion/`
subpackage's `testdata` too).

Two bugs caught while wiring this up, both fixed:
- **Test auto-discovery picking up the abstract base class.** Importing
  `AstGenTest` by name (`from ...ast_gen_test_util import AstGenTest`) put
  it in each concrete test file's module namespace, and `unittest.main()`'s
  default discovery scans *all* `TestCase` subclasses visible in that
  namespace — including the imported base, which has no `PARAM` set and
  crashed with `AttributeError`. Fixed by importing the *module*
  (`from maldoca.astgen.test import ast_gen_test_util`) instead of the
  class, the standard Python idiom for this exact gotcha — only the
  concrete subclass ends up defined in the test file's own namespace.
- **Missing `conversion/` subpackage data.** Same package-boundary issue as
  step 6/12: the 6 IR-enabled cases' golden files under `conversion/` live
  in a separate Bazel package, so the parent directory's `testdata`
  filegroup doesn't include them — needed an explicit second `data` entry
  per case.

Also discovered while transcribing the 9 `INSTANTIATE_TEST_SUITE_P` blocks:
`AstGenTestParam::ir_path` is a plain `std::string` (not
`std::optional<std::string>`) in the C++ struct, so the IR printers
(`PrintIrTableGen`/`PrintAstToIrSource`/`PrintIrToAstSource`) are actually
invoked *unconditionally* by the C++ test for every one of the 9 cases —
`multiple_inheritance`/`typed_lambda` just pass `ir_path=""` (still runs,
produces boilerplate-only output, nothing to compare), and `union` sets a
real `ir_path` but has no IR goldens at all (runs for real, still nothing
to compare). Matched this exactly rather than skipping the IR-printer calls
for cases without goldens, since "does it crash" is itself part of what the
C++ test was checking.

Deleted the `maldoca/astgen/_scratch/` throwaway verification package
(steps 6-13) now that the permanent harness supersedes it.

**Final verification**: `bazel test //maldoca/astgen/...` — all 33 targets
pass (24 original C++ tests, unchanged, plus 9 new Python `ast_gen_test_py`
targets, 72 individual `test_*` methods across them, plus the earlier
`symbol_py_test`/`type_py_test`/`ast_def_py_test`/`printer_base_py_test`/
`cc_printer_base_py_test` unit tests). C++ and Python implementations now
run side by side against the same golden files.

---

## Summary

`maldoca/astgen` has been fully ported from C++ to Python, piece by piece,
with every component verified against the existing test suite before
moving to the next:

| Component | Python file(s) | Verified against |
|---|---|---|
| `Symbol` | `symbol.py` | `symbol_test.py` (ported from `symbol_test.cc`, 7/7) |
| `Type` system | `type.py` | `type_test.py` (ported from `type_test.cc`, 10/10) |
| `AstDef`/`NodeDef`/`FieldDef`/`EnumDef` | `ast_def.py` | `ast_def_test.py` (written from scratch, 14 tests — no C++ equivalent existed) |
| `Printer` (infra) | `printer_base.py` | `printer_base_test.py` (12 tests, written from scratch) |
| `CcPrinterBase` (infra) | `cc_printer_base.py` | `cc_printer_base_test.py` (7 tests, written from scratch) |
| `ast_gen_utils` (infra) | `ast_gen_utils.py` | exercised transitively by every printer test |
| `ast_header_printer` | `ast_header_printer.py` | 9/9 golden `ast.generated.h` |
| `ast_source_printer` | `ast_source_printer.py` | 9/9 golden `ast.generated.cc` |
| `ast_serialize_printer` | `ast_serialize_printer.py` | 9/9 golden `ast_to_json.generated.cc` |
| `ast_from_json_printer` | `ast_from_json_printer.py` | 9/9 golden `ast_from_json.generated.cc` |
| `ts_interface_printer` | `ts_interface_printer.py` | 9/9 golden `ast_ts_interface.generated` |
| `ir_table_gen_printer` | `ir_table_gen_printer.py` | 6/6 golden `<lang>ir_ops.generated.td` |
| `ast_to_ir_source_printer` | `ast_to_ir_source_printer.py` | 6/6 golden `conversion/ast_to_<lang>ir.generated.cc` |
| `ir_to_ast_source_printer` | `ir_to_ast_source_printer.py` | 5/5 golden `conversion/<lang>ir_to_ast.generated.cc` |
| `ast_gen_main` | `ast_gen_main.py` (`py_binary`) | end-to-end run against `test/lambda`, diffed against goldens |

All Python sources live alongside their C++ counterparts in
`maldoca/astgen/` (e.g. `symbol.py` next to `symbol.cc`), with `py_library`/
`py_test`/`py_binary` rules added into the existing `BUILD` files. The C++
implementation has **not** been removed — both implementations currently
coexist, targeting the same golden files, so this is a safe point to pause,
review, or continue toward eventually retiring the C++ side. That decision
(keep both indefinitely vs. delete the C++ implementation once confidence
is high) was intentionally deferred rather than made unilaterally.

Key design decisions, in case they need revisiting:
- Schema stays as `.proto`/`.textproto` (via `py_proto_library`), not
  rewritten — avoids touching the input format or the checked-in test
  inputs.
- `Type`/`AstDef` class hierarchies preserved 1:1 with the C++ originals
  (same inheritance shape), since printers dispatch on concrete type the
  same way `switch (type.kind())` did in C++.
- The bespoke `Printer` class in `printer_base.py` is a deliberate,
  from-scratch reimplementation of the *subset* of
  `google::protobuf::io::Printer` actually used (`$var$` substitution,
  scoped indent/vars) — not a general-purpose port of the upstream type.
- No abseil-py / third-party CLI framework dependency introduced;
  `ast_gen_main.py` uses stdlib `argparse` since nothing else in this repo
  pulls in abseil-py and there was no reason to be the first.

### 2026-07-04 — Rebased onto test_942383408; verified against the real maldoca/js AST

Rebased `astgen-python` onto `test_942383408`, which added the real,
production JS AST schema (`maldoca/js/ast/ast_def.textproto`, ~3400 lines)
plus a C++ `ast_gen_test` comparing it against the real checked-in
`maldoca/js/ast/*.generated.*` / `maldoca/js/ir/*.generated.*` /
`maldoca/js/ir/conversion/*.generated.cc` files — this is the schema that,
per the step-0 investigation notes, wasn't available anywhere in the repo
before. Rebase was clean (no conflicts): the two branches touched disjoint
files.

That commit also merged HIR into IR (deprecated `NodeDef::has_control_flow`
for dialect-naming purposes — no more `<lang>hir` vs `<lang>ir` split, only
`<lang>ir`), editing `ast_def.cc/h/proto` and `ir_table_gen_printer.cc`.
Updated the two Python files with the equivalent logic
(`ast_def.py::NodeDef.ir_op_name()`, `ir_table_gen_printer.py::print_node()`)
by grepping for every Python reference to `has_control_flow`/`hir_name`/
`HirName` first to confirm the change's exact scope, rather than guessing —
found exactly the two files the C++ diff touched, nothing more. None of the
9 small test schemas set `has_control_flow`, so their goldens didn't need
regenerating (confirmed via `bazel test //maldoca/astgen/...` — all still
pass unchanged).

**Verified the Python port against the real production schema**: added
`maldoca/js/ast/ast_gen_test.py` (mirroring the new
`maldoca/js/ast/ast_gen_test.cc`) plus the necessary `py_test`/`filegroup`
BUILD plumbing (`maldoca/js/ast/BUILD`, and two small `filegroup`s in
`maldoca/js/ir/BUILD` / `maldoca/js/ir/conversion/BUILD` to expose
`jsir_ops.generated.td` and the `ast_to_jsir`/`jsir_to_ast` generated
sources as depend-able targets, since they're otherwise only consumed as
`td_library`/`cc_library` srcs). First ran the *C++* `ast_gen_test` as a
baseline to confirm the checked-in generated files really are in sync with
this schema (they are), then ran the Python equivalent.

**Result: `bazel test //maldoca/js/ast:ast_gen_test_py` — 8/8 pass.** The
from-scratch Python rewrite produces byte-for-byte identical output to the
real `ast.generated.h`, `ast.generated.cc`, `ast_to_json.generated.cc`,
`ast_from_json.generated.cc`, `jsir_ops.generated.td`,
`ast_to_jsir.generated.cc`, and `jsir_to_ast.generated.cc` — the actual
files powering the production JSIR compiler, not just the 9 synthetic test
schemas. Full suite (`//maldoca/astgen/...` + the two `maldoca/js/ast`
targets): 35/35 pass.
