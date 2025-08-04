# Intermediate Representation Design

## Overview

JSIR is a next-generation JavaScript analysis tool from Google. At its core
is an [MLIR](https://mlir.llvm.org)-based high-level intermediate representation
(IR). More specifically:

*   JSIR **retains all information** from the ([Babel](https://babeljs.io))
    abstract syntax tree (AST), including original control flow structures,
    source ranges, comments, etc.. As a result, **JSIR can be fully lifted back
    to the AST**, and therefore back to the source, making it suitable for
    source-to-source transformation.

*   JSIR provides a standard **dataflow analysis API**, built on top of the
    upstream [MLIR API](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis),
    with ease-of-use improvements.

This design was driven by the diverse needs at Google for malicious JavaScript
analysis and detection. For example, taint analysis is a dataflow analysis which
requires a CFG; decompilation requires lifting low-level representations to
source code; deobfuscation is a source-to-source transformation.

To achieve the design goals, JSIR defines two MLIR dialects:

*   **JSHIR:** A high-level IR that has a nearly one-to-one mapping with the
    abstract syntax tree (AST). JSHIR models control flow structures using MLIR
    [regions](https://mlir.llvm.org/docs/LangRef/#regions).

*   **JSLIR:** A low-level IR that uses Control Flow Graphs (CFGs) to represent
    branching behaviors. JSLIR adds extra annotations to the CFGs, to preserve
    information about original control flow structures and enable full
    conversion back to the JSHIR.

## Achieving AST ↔ IR Roundtrip

A critical goal of this project is to ensure an accurate lift of the IR back to
the AST. This "reversible" IR design enables source-to-source transformations -
we perform IR transformations then lift the transformed IR to source.

Internal evaluations on billions of JavaScript samples showed that, AST - IR
round-trips achieved 99.9%+ success resulting in the same source.

In the following sections, we will describe important design decisions that
achieve this perfect round-trip.

## Post-order traversal of AST

Let’s start from the simplest case - straightline code, i.e. a list of
statements with no control flow structures like `if`-statements.

Each of these simple expression / statement AST node is mapped to a
corresponding JSIR operation. Therefore, JSIR for straightline code is
equivalent to a post-order traversal dump of the AST.

For example, for the following JavaScript statements:

```js
1 + 2 + 3;
4 * 5;
```

The corresponding AST is as follows (see
[astexplorer](https://astexplorer.net/#/gist/8de510a68663424455bb9c175698cd38/f3e6d96bfe1bfa8ab11783eae0e2e7e22209ece9)
for the full AST):

```c++
[
  ExpressionStatement {
    expression: BinaryExpression {
      op: '+',
      left: BinaryExpression {
        op: '+',
        left: NumericLiteral { value: 1 }
        right: NumericLiteral { value: 2 }
      }
      right: NumericLiteral { value: 3 }
    }
  },
  ExpressionStatement {
    expression: BinaryExpression {
      op: '*',
      left: NumericLiteral { value: 4 }
      right: NumericLiteral { value: 5 }
    }
  },
]
```

The corresponding JSIR is as follows:

```c++
%1 = jsir.numeric_literal {1}
%2 = jsir.numeric_literal {2}
%1_plus_2 = jsir.binary_expression {'+'} (%1, %2)
%3 = jsir.numeric_literal {3}
%1_plus_2_plus_3 = jsir.binary_expression {'+'} (%1_plus_2, %3)
jsir.expression_statement (%1_plus_2_plus_3)
%4 = jsir.numeric_literal {4}
%5 = jsir.numeric_literal {5}
%4_mult_5 = jsir.binary_expression {'*'} (%4, %5)
jsir.expression_statement (%4_mult_5)
```

To lift this IR back to the AST, we **cannot** treat each op as a separate
statement, because that would cause every SSA value (e.g. `%1`) to become a
local variable:

```js {.bad}
// Too many local variables!
var $1 = 1;
var $2 = 2;
var $1_plus_2 = $1 + $2;
var $3 = 3;
var $1_plus_2_plus_3 = $1_plus_2 + $3;
$1_plus_2_plus_3;  // jsir.expression_statement
var $4 = 4;
var $5 = 5;
var $4_mult_5 = $4 * $5;
$4_mult_5;  // jsir.expression_statement
```

However, we can detect the two statement-level ops (i.e. the two
`jsir.expression_statement` ops) and recursively traverse their use-def chains:

```js {.good}
   1 + 2 + 3 ;
// ~           %1 = jsir.numeric_literal {1}
//     ~       %2 = jsir.numeric_literal {2}
// ~~~~~       %1_plus_2 = jsir.binary_expression {'+'} (%1, %2)
//         ~   %3 = jsir.numeric_literal {3}
// ~~~~~~~~~   %1_plus_2_plus_3 = jsir.binary_expression {'+'} (%1_plus_2, %3)
// ~~~~~~~~~~~ jsir.expression_statement (%1_plus_2_plus_3)

   4 * 5 ;
// ~       %4 = jsir.numeric_literal {4}
//     ~   %5 = jsir.numeric_literal {5}
// ~~~~~   %4_mult_5 = jsir.binary_expression {'*'} (%4, %5)
// ~~~~~~~ jsir.expression_statement (%4_mult_5)
```

When we try to lift a basic block (`mlir::Block`) of JSIR ops we always know
ahead of time what "kind" of content it holds:

*   If the block holds **a statement**, then we find the single statement-level
    op and traverse its use-def chain to generate a `JsStatement` AST node.

*   If the block holds **a list of statements**, then we find all the
    statement-level ops and traverse their use-def chains to generate a list of
    `JsStatement` AST nodes.

*   If the block holds **an expression**, then it always ends with a
    `jsir.expr_region_end (%expr)` op. We traverse the use-def chain of `%expr`
    to generate a `JsExpression` AST node.

*   If the block holds **a list of expressions**, then it always ends with a
    `jsir.exprs_region_end (%e1, %e2, ...)` op. We traverse the use-def chains
    of `%e1, %e2, ...` to generate a list of `JsExpression` AST nodes.

## Symbols, l-values and r-values

We distinguish between lvalues and rvalues in JSIR. For example, consider the
following assignment:

```js
a = b;
```

`a` is an lvalue, and `b` is an rvalue.

L-values ane r-values are represented in the **same** way in the AST:

```c++
ExpressionStatement {
  expression: AssignmentExpression {
    left: Identifier {"a"}
    right: Identifier {"b"}
  }
}
```

However, they are represented **differently** in the IR:

```c++
%a_ref = jsir.identifier_ref {"a"}  // lvalue
%b = jsir.identifier {"b"}          // rvalue
%assign = jsir.assignment_expression (%a_ref, %b)
jsir.expression_statement (%assign)
```

The reason for this distinction is to explicitly represent the different
semantic meanings:

*   An l-value is a reference to some object / some memory location;

*   An rvalue is some value.

## Representing control flows: JSHIR

As mentioned above, JSHIR seeks to have a nearly one-to-one mapping from the
AST. Therefore, to preserve all information about the original control flow
structures, we define a separate op for each control flow structure (e.g.
`jshir.if_statement`, `jshir.while_statement`, etc.). The nested code blocks are
represented as MLIR [regions](https://mlir.llvm.org/docs/LangRef/#regions).

For example, consider the following `if`-statement:

```js
if (cond)
  a;
else
  b;
```

Its corresponding AST is as follows:

```c++
IfStatement {
  test: Identifier {"cond"}
  consequent: ExpressionStatement {
    expression: Identifier {"a"}
  }
  alternate: ExpressionStatement {
    expression: Identifier {"b"}
  }
}
```

And, its corresponding JSHIR is as follows:

```mlir
%cond = jsir.identifier {"cond"}
jshir.if_statement (%cond) ({
  %a = jsir.identifier {"a"}
  jsir.expression_statement (%a)
}, {
  %b = jsir.identifier {"b"}
  jsir.expression_statement (%b)
})
```

Since nested structure is fully preserved, lifting JSHIR back to the AST is
achieved by a standard recursive traversal.

## Representing control flows: JSLIR

### Control flow graph

The region-based nested structure of JSHIR, though intuitive and readable, does
not provide enough control flow information to facilitate
[dataflow analysis](https://en.wikipedia.org/wiki/Data-flow_analysis).

Dataflow analysis traverses the IR as a graph, which requires knowing the "next
op" of each op. Ironically, a program with only `goto`-statements but no
structured control flow would satisfy this requirement. IRs typically lower
structured control flow to branch ops which are essentially `goto`-statements.
This is also what JSLIR does.

For example, for the following JSHIR:

```mlir
%cond = jsir.identifier {"cond"}
jshir.if_statement (%cond) ({
  %a = jsir.identifier {"a"}
  jsir.expression_statement (%a)
}, {
  %b = jsir.identifier {"b"}
  jsir.expression_statement (%b)
})
```

We lower the `jshir.if_statement` into the following CFG with branch ops:

<pre><code>  %cond = jsir.identifier {"cond"}
  <b>// if %cond goto ^bb_true else goto ^bb_false</b>
  <b>cf.cond_br (%cond) [^bb_true, ^bb_false]</b>

^bb_true:
  %a = jsir.identifier {"a"}
  jsir.expression_statement (%a)
  <b>// goto ^bb_end</b>
  <b>cf.br [^bb_end]</b>

^bb_false:
  %b = jsir.identifier {"b"}
  jsir.expression_statement (%b)
  <b>// goto ^bb_end</b>
  <b>cf.br [^bb_end]</b>

^bb_end:
  ...
</code></pre>

In particular:

*   We flatten the nested structure into blocks (e.g. `^bb_true`);

*   Each block contains a list of ops;

*   We explicitly represent branch behaviors with `cf.cond_br`
    ("control_flow.conditional_branch") and `cf.br` ("control_flow.branch") ops.
    In particular, at the entry of the `if`-statement, we branch into either
    `bb_true` or `bb_false` based on the condition `%cond`; and at the end of
    both blocks we branch and merge to the same block `bb_end`.

### Challenge: lift back to AST

However, this IR would lose the explicit information that these blocks represent
an `if`-statement, which makes it hard to lift back the AST.

This challenge is more evident with more complex control flow structures. For
example, consider the following code with an `if`-statement within a
`while`-statement.

```js
...
while (cond)
  if (test)
    body;
...
```

The JSLIR is as follows:

``` {.bad}
  ...
  cf.br [^bb1]

^bb1:
  %cond = jsir.identifier {"cond"}
  cf.cond_br (%cond) [^bb2, ^bb5]

^bb2:
  %test = jsir.identifier {"test"}
  cf.cond_br (%test) [^bb3, ^bb4]

^bb3:
  %body = jsir.identifier {"body"}
  jsir.expression_statement (%body)
  cf.br [^bb4]

^bb4:
  cf.br [^while_test]

^bb5:
  ...
```

### Solution: annotation ops

Our solution is to add additional annotation ops to JSLIR which provide explicit
information about what control flow structure each block represents. These
annotation ops do not affect the semantics of the IR (i.e. they are considered
no-ops during analyses). However, during the lift, we can use them to guide a
recursive traversal of the original control flow structures.

For example, the annotated JSLIR for the `if`-statement is as follows:

<pre><code>  %cond = jsir.identifier {"cond"}
  <b>%token = jslir.control_flow_starter {IfStatement}</b>
  cf.cond_br (%cond) [^bb_true, ^bb_false]

^bb_true:
  <b>jslir.control_flow_marker (%token) {IfStatementConsequent}</b>
  %a = jsir.identifier {"a"}
  jsir.expression_statement (%a)
  cf.br [^bb_end]

^bb_false:
  <b>jslir.control_flow_marker (%token) {IfStatementAlternate}</b>
  %b = jsir.identifier {"b"}
  jsir.expression_statement (%b)
  cf.br [^bb_end]

^bb_end:
  <b>jslir.control_flow_marker (%token) {IfStatementEnd}</b>
  ...
</code></pre>

Note that by tracing the uses of `%token`, we can locate the starting points of
all components of an `if`-statement.
