# Intermediate Representation Design

## Overview

JSIR is a next-generation JavaScript analysis tooling from Google. At its core
is an [MLIR](https://mlir.llvm.org)-based high-level intermediate representation
(IR). More specifically:

*   JSIR **retains all information** from the ([Babel](https://babeljs.io))
    abstract syntax tree (AST), including original control flow structures,
    source ranges, comments, etc. As a result, **JSIR can be fully lifted back
    to the AST**, and therefore back to the source, making it suitable for
    source-to-source transformation.

*   JSIR provides a standard **dataflow analysis API**, built on top of the
    built-in [MLIR API](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis),
    with ease-of-use improvements.

This design was driven by the diverse needs at Google for malicious JavaScript
analysis and detection. For example, taint analysis is a dataflow analysis;
decompilation requires lifting low-level representations to source code;
deobfuscation is a source-to-source transformation.

To achieve these design goals, JSIR is designed as a high-level IR that has a
nearly one-to-one mapping with the abstract syntax tree (AST), and models
control flow structures using MLIR
[regions](https://mlir.llvm.org/docs/LangRef/#regions).

## Achieving AST ↔ IR Roundtrip

A critical goal of JSIR is to ensure an accurate lift of the IR back to the AST.
This "reversible" IR design enables source-to-source transformations - we
perform IR transformations then lift the transformed IR to source.

Internal evaluations on billions of JavaScript samples showed that AST - IR
round-trips achieved 99.9%+ success resulting in the same source.

In the following sections, we will describe important design decisions that
achieve this high-fidelity round-trip.

## Post-order traversal of AST

Let’s start from the simplest case - straight-line code, i.e. a list of
statements with no control flow structures like `if`-statements.

Each of these simple expression / statement AST nodes is mapped to a
corresponding JSIR operation. Therefore, JSIR for straight-line code is
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
        left: NumericLiteral { value: 1 },
        right: NumericLiteral { value: 2 }
      },
      right: NumericLiteral { value: 3 }
    }
  },
  ExpressionStatement {
    expression: BinaryExpression {
      op: '*',
      left: NumericLiteral { value: 4 },
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

Perhaps the one-to-one mapping from AST nodes to JSIR operations is more obvious
if we add some indentations:

```mlir
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
// ~                 %1 = jsir.numeric_literal {1}
//     ~             %2 = jsir.numeric_literal {2}
// ~~~~~           %1_plus_2 = jsir.binary_expression {'+'} (%1, %2)
//         ~       %3 = jsir.numeric_literal {3}
// ~~~~~~~~~     %1_plus_2_plus_3 = jsir.binary_expression {'+'} (%1_plus_2, %3)
// ~~~~~~~~~~~ jsir.expression_statement (%1_plus_2_plus_3)

   4 * 5 ;
// ~               %4 = jsir.numeric_literal {4}
//     ~           %5 = jsir.numeric_literal {5}
// ~~~~~         %4_mult_5 = jsir.binary_expression {'*'} (%4, %5)
// ~~~~~~~     jsir.expression_statement (%4_mult_5)
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

We distinguish between l-values and r-values in JSIR. For example, consider the
following assignment:

```js
a = b;
```

`a` is an l-value, and `b` is an r-value.

L-values and r-values are represented in the **same** way in the AST:

```c++
ExpressionStatement {
  expression: AssignmentExpression {
    left: Identifier {"a"},
    right: Identifier {"b"}
  }
}
```

However, they are represented **differently** in the IR:

```c++
%a_ref = jsir.identifier_ref {"a"}  // l-value
%b = jsir.identifier {"b"}          // r-value
%assign = jsir.assignment_expression (%a_ref, %b)
jsir.expression_statement (%assign)
```

The reason for this distinction is to explicitly represent the different
semantic meanings:

*   An l-value is a reference to some object / some memory location;

*   An r-value is some value.

> **NOTE:** We will likely revisit how we represent symbols.

## Representing control flows

As mentioned above, JSIR seeks to have a nearly one-to-one mapping from the
AST. Therefore, to preserve all information about the original control flow
structures, we define a separate op for each control flow structure (e.g.
`jshir.if_statement`, `jshir.while_statement`, etc.). The nested code blocks are
represented as MLIR [regions](https://mlir.llvm.org/docs/LangRef/#regions).

### Example: `if`-statement

Consider the following `if`-statement:

```js
if (cond)
  a;
else
  b;
```

Its corresponding AST is as follows
([astexplorer](https://astexplorer.net/#/gist/58e3ca121e8bc97d9d1987766f4df96a/37b0de0e94073d24f40aede05b14e1c480b7b39a)):

```c++
IfStatement {
  test: Identifier { name: "cond" },
  consequent: ExpressionStatement {
    expression: Identifier { name: "a" }
  },
  alternate: ExpressionStatement {
    expression: Identifier { name: "b" }
  }
}
```

And, its corresponding JSIR is as follows:

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

Since nested structure is fully preserved, lifting JSIR back to the AST is
achieved by a standard recursive traversal.

### Example: `while`-statement

Consider the following `while`-statement:

```js
while (cond())
  x++;
```

Its corresponding AST is as follows ([astexplorer](https://astexplorer.net/#/gist/58e3ca121e8bc97d9d1987766f4df96a/6ce08d84210afbacdf99732366e04eafcd6b3ab5)):

```c++
WhileStatement {
  test: CallExpression {
    callee: Identifier { name: "cond" },
    arguments: []
  },
  body: ExpressionStatement {
    expression: UpdateExpression {
      operator: "++",
      prefix: false,
      argument: Identifier { name: "x" }
    }
  }
}
```

Its corresponding JSIR is as follows:

```mlir
jshir.while_statement ({
  %cond_id = jsir.identifier {"cond"}
  %cond_call = jsir.call_expression (%cond_id)
  jsir.expr_region_end (%cond_call)
}, {
  %x_ref = jsir.identifier_ref {"x"}
  %update = jsir.update_expression {"++"} (%x_ref)
  jsir.expression_statement (%update)
})
```

Note that unlike `jshir.if_statement`, the condition in a
`jshir.while_statement` is represented as a region rather than a normal SSA
value (`%cond`). This is because the condition is evaluated in each iteration
**within** the `while`-statement, whereas the condition is evaluated only once
**before** the `if`-statement.

### Example: logical expression

Consider the following statement with a logical expression:

```js
x = a && b;
```

Its corresponding AST is as follows ([astexplorer](https://astexplorer.net/#/gist/58e3ca121e8bc97d9d1987766f4df96a/c7fbec034a61bcbb66959714b7d95dbd9ca86e32)):

```c++
ExpressionStatement {
  expression: AssignmentExpression {
    left: Identifier { name: "x" },
    right: LogicalExpression {
      left: Identifier { name: "a" },
      right: Identifier { name: "b" }
    }
  }
}
```

Its corresponding JSIR is as follows:

```mlir
%x_ref = jsir.identifier_ref {"x"}
%a = jsir.identifier {"a"}
%and = jshir.logical_expression (%a) ({
  %b = jsir.identifier {"b"}
  jsir.expr_region_end (%b)
})
%assign = jsir.assignment_expression (%x_ref, %and)
jsir.expression_statement (%assign)
```

Note that in `jshir.logical_expression`, `left` is an SSA value, and `right` is
a region. This is because `left` is always evaluated first, whereas `right` is
only evaluated if the result of `left` is truthy, and omitted if `left` is falsy
due to the short-circuit behavior.
