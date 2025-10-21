# Dataflow Analysis

## Overview

JSIR provides an API for **flow-sensitive, conditional**, dataflow analysis.

It is built on top of the MLIR dataflow analysis API with usability
improvements:

*   We allow a single analysis to use sparse and dense states at the same time,
    whereas a user of the MLIR API would have to write two separate analyses.

*   We have a util API for updating analysis states that automatically trigger
    op visits, whereas a user of the MLIR API would have to manually call
    `propagateIfChanged()` to propagate states.

#### Flow-sensitive analysis

**Flow-sensitive** means for the following code, we can reason about the two
program paths, and determine that, at the end of the `if`-statement, `a` has the
value `“same”`, and `b` has an unknown value.

```js
if (some_unknown_condition) {
  a = "same";
  b = "different 1";
} else {
  a = "same";
  b = "different 2";
}
// a == "same"
// b == <Unknown>
```

#### Conditional analysis

**Conditional** means for the following code, we can skip the traversal of the
`else`-branch of the `if`-statement and determine that at the end of the
`if`-statement, `b` has the value `“different 1”`.

```js
if (true) {
  a = "same";
  b = "different 1";
} else {
  // dead code
  a = "same";
  b = "different 2";
}
// a == "same"
// b == "different 1"
```

<!--  TODO: Explain how the actual API looks like in C++. -->

## Example 1: Straightline code

> In this example, we demonstrate the fundamental concepts of dataflow analysis:
>
> *   **Program points** and **values** - where we attach states;
>
> *   **Transfer function** - what computes the states;
>
> *   **Constant propagation** - how these concepts are used in action.

Consider running constant propagation on the following code:

```js
a = 1;
b = a + 2;
```

As shown below, the analysis calculates the state at each program point, which
contains the value held by each variable. In particular, at the end of the
analysis, we expect to determine that `b` has the value of `3` after the
assignment.

```js
// {}
a = 1;
// {a: Const{1}}
b = a + 2;
// {a: Const{1}, b: Const{3}}
```

Now let's see how this works in detail. The dataflow analysis algorithm runs on
JSIR, as shown below:

```
// IR for `a = 1;`:
%a_ref = jsir.identifier_ref{"a"}
%1 = jsir.numeric_literal{1}
%a_assign = jsir.assignment_expression (%a_ref, %1)
jsir.expression_statement (%a_assign)

// IR for `b = a + 2;`
%b_ref = jsir.identifier_ref{"b"}
%a = jsir.identifier{"a"}
%2 = jsir.numeric_literal{2}
%add = jsir.binary_expression{"+"} (%a, %2)
%b_assign = jsir.assignment_expression (%b_ref, %add)
jsir.expression_statement (%b_assign)
```

As we can see, JSIR in this example is essentially a **post-order traversal** of
the abstract syntax tree (AST). This is intentional - we designed JSIR to be as
close to the AST as possible.

The dataflow analysis algorithm will compute a **state** attached to every
**program point** and every **value**.

> ### Concepts: program points and values
>
> *   A **program point** is before or after each MLIR operation
>     (`mlir::Operation`). In this example, they are P0 through P10.
>
> *   A **value** is an MLIR SSA value (`mlir::Value`), denoted by `%value`.
>
> In constant propagation, we design our **value state** as either `Uninit`,
> `Const{...}`, or `Unknown`, and **program point state** as a map from symbols
> to value states.

We expect that, at the end of the analysis, the states on all program points and
values should be as follows:

```
// state[P0] = {}

%a_ref = jsir.identifier_ref{"a"}
// state[%a_ref] = Uninit
// state[P1] = {}

%1 = jsir.numeric_literal{1}
// state[%1] = Const{1}
// state[P2] = {}

%a_assign = jsir.assignment_expression (%a_ref, %1)
// state[%a_assign] = Const{1}
// state[P3] = {a: Const{1}}

jsir.expression_statement (%a_assign)
// state[P4] = {a: Const{1}}

%b_ref = jsir.identifier_ref{"b"}
// state[%b_ref] = Uninit
// state[P5] = {a: Const{1}}

%a = jsir.identifier{"a"}
// state[%a] = Const{1}
// state[P6] = {a: Const{1}}

%2 = jsir.numeric_literal{2}
// state[%2] = Const{2}
// state[P7] = {a: Const{1}}

%add = jsir.binary_expression{"+"} (%a, %2)
// state[%add] = Const{3}
// state[P8] = {a: Const{1}}

%b_assign = jsir.assignment_expression (%b_ref, %add)
// state[%b_assign] = Const{3}
// state[P9] = {a: Const{1}, b: Const{3}}

jsir.expression_statement (%b_assign)
// state[P10] = {a: Const{1}, b: Const{3}}
```

The dataflow analysis algorithm traverses the IR and computes all these states.

(We will defer the discussion on the traversal order to
[example 2](#example-2-if-statement), where we have an IR that's more complex
than straightline code.)

During the traversal, when we visit an operation, we will:

*   **Read** the state on the program point **before** the op, and states on its
    **input values**; and
*   **Write** the state on the program point **after** the op, and states on its
    **output values**.

> **Example 1.1:** When we visit the `jsir.binary_expression` op, we will:
>
> *   Read the state before the op, i.e. `state[P7]`, and the states on its
>     input values `%a` and `%2`; and
>
> *   Write the state after the op, i.e. `state[P8]`, and the state on its
>     output value `%add`.
>
> These are all the reads and writes:
>
> ```
> // READ:  state[P7]   == {a: Const{1}}
> // READ:  state[%a]  == Const{1}
> // READ:  state[%2]  == Const{2}
> %add = jsir.binary_expression{"+"} (%a, %2)
> // WRITE: state[P8]    = {a: Const{1}}
> // WRITE: state[%add] = Const{3}
> ```

> **Example 1.2:** When we visit the `jsir.assignment_expression` op, we will:
>
> *   Read the state before the op, i.e. `state[P8]`, and the states on its
>     input values `%b_ref` and `%add`; and
> *   Write the state after the op, i.e. `state[P9]`, and the state on its
>     output value `%b_assign`.
>
> These are all the reads and writes:
>
> ```
> // READ: state[P8]         == {a: Const{1}}
> // READ: state[%b_ref]    == Uninit
> // READ: state[%add]      == Const{3}
> %b_assign = jsir.assignment_expression (%b_ref, %add)
> // WRITE: state[P9]         = {a: Const{1}, b: Const{3}}
> // WRITE: state[%b_assign] = Const{3}
> ```

> **Caveat:** The lvalue argument of `jsir.assignment_expression`, such as
> `%b_ref`, is not a value in terms of constant propagation, as it represents a
> reference to a variable, so its state is `Uninit`. When we visit the
> `jsir.assignment_expression` op, we fetch the defining op of `%b_ref` (i.e.
> the `jsir.identifier_ref` op) to get the variable name it refers to. We intend
> to improve our API to handle lvalues in a better way.

When we have completed the calculation of every state in the IR, the algorithm
terminates.

## Example 2: If statement

An `if`-statement is a control flow structure, which means the IR is no longer a
piece of straight-line code. Through this example, we will demonstrate:

*   **MLIR regions** - how control flow structures are represented in JSIR

*   **CFG edges** - how branch behaviors are modeled during dataflow analysis

*   **Work queue** - how the IR traversal order is determined during dataflow
    analysis

*   **Lattices** - how we design states

Consider the following `if`-statement:

```js
...
// We don't know what cond is.
if (cond)
  a = 1;
else
  a = 2;
// a == Unknown
...
```

Let's assume that we don’t know the value of `cond`, so we don’t know which
branch we will take. Therefore, we don't know what `a` will be after the
`if`-statement. We expect that the constant propagation analysis will determine
that `a` is `Unknown` after the `if`-statement.

The JSIR for the code above is as follows:

```
...
%cond = jsir.identifier{"cond"}
jshir.if_statement (%cond) ({
  %a_ref_true = jsir.identifier_ref{"a"}
  %1 = jsir.numeric_literal{1}
  %assign_true = jsir.assignment_expression (%a_ref_true, %1)
  jsir.expression_statement (%assign_true)
}, {
  %a_ref_false = jsir.identifier_ref{"a"}
  %2 = jsir.numeric_literal{2}
  %assign_false = jsir.assignment_expression (%a_ref_false, %2)
  jsir.expression_statement (%assign_false)
})
...
```

In JSIR, we represent control flow structures using **MLIR regions**. This
preserves the nested structures. We can see that the two branches of the
`if`-statement are represented by two regions.

> **Caveat:** We are omitting the fact that the ops are actually not directly
> stored in regions, but in blocks. In particular, each of the two regions
> contains a single block. This is not important in JSIR since we never store
> more than 1 block in any region.

The JSIR dataflow analysis API understands the branching behaviors of
`jshir.if_statement`, and builds **CFG (control flow graph) edges** to represent
them internally:

```
      ...
      %cond = jsir.identifier{"cond"}
┌─────◄
│     jshir.if_statement (%cond) ({
├───────►
│       %a_ref_true = jsir.identifier_ref{"a"}
│       %1 = jsir.numeric_literal{1}
│       %assign_true = jsir.assignment_expression (%a_ref_true, %1)
│       jsir.expression_statement (%assign_true)
│  ┌────◄
│  │  }, {
└──│────►
   │    %a_ref_false = jsir.identifier_ref{"a"}
   │    %2 = jsir.numeric_literal{2}
   │    %assign_false = jsir.assignment_expression (%a_ref_false, %2)
   │    jsir.expression_statement (%assign_false)
   ├────◄
   │  })
   └──►
      ...
```

> ### Concept: CFG Edge
>
> A CFG edge represents a branch from one program point to another.
>
> The name resembles the fact that we can view the IR as a control flow graph,
> where program points are nodes, and ops and CFG edges are, well, edges that
> connect these nodes. The dataflow analysis algorithm is essentially performing
> a graph traversal.

### Step 1: Before the `if`-statement

The analysis starts at the top, and we first compute states as usual, until
reaching `B1`, i.e. right **B**efore the `if`-statement.

<pre><code>      ...
      <b>// state[B0] = {}</b>
      %cond = jsir.identifier{"cond"}
      <b>// state[%cond] = Unknown</b>
      <b>// state[B1] = {}</b>
┌─────◄
│     jshir.if_statement (%cond) ({
├───────►
│       &lt;IR for `a = 1;`&gt;
│  ┌────◄
│  │  }, {
└──│────►
   │    &lt;IR for `a = 2;`&gt;
   ├────◄
   │  })
   └──►
      ...
</code></pre>

### Step 2: Propagating the state to both branches

Now we trace the two CFG edges originating from `B1` and propagate `state[B1]`
to two program points: `T0` and `F0`, which represent the entry points of the
`true`-branch and the `false`-branch.

<pre><code>      ...
      // state[B0] = {}
      %cond = jsir.identifier{"cond"}
      // state[%cond] = Unknown
      // state[B1] = {}
┌─────◄
│     jshir.if_statement (%cond) ({
├───────►
│       <b>// state[T0] = {}</b>
│       &lt;IR for `a = 1;`&gt;
│  ┌────◄
│  │  }, {
└──│────►
   │    <b>// state[F0] = {}</b>
   │    &lt;IR for `a = 2;`&gt;
   ├────◄
   │  })
   └──►
      ...
</code></pre>

This propagation is organized by the **work queue**.

> ### Concept: the work queue
>
> In the dataflow analysis algorithm, whenever we modify the state at a
> **program point**, we will trigger the visits of all **IR objects** that
> depend on it. For example, when we modify `state[B0]`, we will trigger the
> visit of the op `%cond = jsir.identifier{"cond"}`, since this op depends on
> `state[B0]` as input.
>
> More specifically, we maintain a **work queue** of **IR objects** to be
> visited. Modifying a state will insert its dependent IR objects into the work
> queue.
>
> **Caveat:** (This is an implementation detail that doesn't affect the user of
> the dataflow analysis API.) Conceptually speaking, when we modify `state[B1]`,
> we should insert the two CFG edges into the work queue. However, due to lack
> of flexibility in the underlying MLIR API, our CFG edges are not a supported
> type of "visitee". Therefore, we insert the **destinations** of the CFG edges,
> i.e. `T0` and `F0` into the work queue, and simulate the visits of the CFG
> edges when visiting them. We hope to refactor the MLIR API to fix this.

### Step 3: Computing states in the `true`-branch

Now let's focus on the `true`-branch. Since we modified `state[T0]` due to the
propagation before, we will consequently visit all ops in the region.

> **Caveat:** In practice, due to the work queue, the traversals of the two
> regions should be interleaved. However, the order doesn't affect the end
> result of the algorithm anyway. Therefore, here we "pretend" to traverse the
> `true`-branch first.

<pre><code>      ...
      // state[B0] = {}
      %cond = jsir.identifier{"cond"}
      // state[%cond] = Unknown
      // state[B1] = {}
┌─────◄
│     jshir.if_statement (%cond) ({
├───────►
│       // state[T0] = {}
│       %a_ref_true = jsir.identifier_ref{"a"}
│       <b>// state[%a_ref_true] = Uninit</b>
│       <b>// state[T1] = {}</b>
│       %1 = jsir.numeric_literal{1}
│       <b>// state[%1] = Const{1}</b>
│       <b>// state[T2] = {}</b>
│       %assign_true = jsir.assignment_expression (%a_ref_true, %1)
│       <b>// state[%assign_true] = Const{1}</b>
│       <b>// state[T3] = {a: Const{1}}</b>
│       jsir.expression_statement (%assign_true)
│       <b>// state[T4] = {a: Const{1}}</b>
│  ┌────◄
│  │  }, {
└──│────►
   │    // state[F0] = {}
   │    &lt;IR for `a = 2;`&gt;
   ├────◄
   │  })
   └──►
      ...
</code></pre>

Traversing the `true`-branch will cause `state[T4]` to be evaluated to `{a:
Const{1}}`.

### Step 4: Propagating out of the `true`-branch

Now that we have reached `T4`, i.e. the end of the `true`-branch, we will follow
the CFG edge and propagate its state to `E0`, i.e. the end of the
`if`-statement.

<pre><code>      ...
      // state[B0] = {}
      %cond = jsir.identifier{"cond"}
      // state[%cond] = Unknown
      // state[B1] = {}
┌─────◄
│     jshir.if_statement (%cond) ({
├───────►
│       // state[T0] = {}
│       &lt;IR for `a = 1;`&gt;
│       // state[T4] = {a: Const{1}}
│  ┌────◄
│  │  }, {
└──│────►
   │    // state[F0] = {}
   │    &lt;IR for `a = 2;`&gt;
   ├────◄
   │  })
   └──►
      <b>// state[E0] = {a: Const{1}}</b>
      ...
</code></pre>

> **Important:** Since we don’t know which branch in the if-statement we will
> take, we shouldn't know the value of `a` is after the if-statement. However,
> right now `state[E0]` says `a` has constant `1`, which is incorrect. This is
> fine - this is an intermediate state in the dataflow analysis algorithm.

### Step 5: Computing states in the `false`-branch

From the work queue, we also need to traverse the `false`-branch, which will
compute states up to `state[F4]`:

<pre><code>      ...
      // state[B0] = {}
      %cond = jsir.identifier{"cond"}
      // state[%cond] = Unknown
      // state[B1] = {}
┌─────◄
│     jshir.if_statement (%cond) ({
├───────►
│       // state[T0] = {}
│       &lt;IR for `a = 1;`&gt;
│       // state[T4] = {a: Const{1}}
│  ┌────◄
│  │  }, {
└──│────►
   │    // state[F0] = {}
   │    %a_ref_false = jsir.identifier_ref{"a"}
   │    <b>// state[%a_ref_false] = Uninit</b>
   │    <b>// state[F1] = {}</b>
   │    %2 = jsir.numeric_literal{2}
   │    <b>// state[%2] = Const{2}</b>
   │    <b>// state[F2] = {}</b>
   │    %assign_false = jsir.assignment_expression (%a_ref_false, %2)
   │    <b>// state[%assign_false] = Const{2}</b>
   │    <b>// state[F3] = {a: Const{2}}</b>
   │    jsir.expression_statement (%assign_false)
   │    <b>// state[F4] = {a: Const{2}}</b>
   ├────◄
   │  })
   └──►
      // state[E0] = {a: Const{1}}
      ...
</code></pre>

### Step 6: Joining the two paths!

Now we have reached `F4`, the end of the `false`-branch. Similarly, we should
propagate its state to `E0`.

**Important:** We don't simply overwrite `state[E0]` with `state[F4]`, but we
**join `state[F4]` into `state[E0]`**:

<pre><code>      ...
      // state[B0] = {}
      %cond = jsir.identifier{"cond"}
      // state[%cond] = Unknown
      // state[B1] = {}
┌─────◄
│     jshir.if_statement (%cond) ({
├───────►
│       // state[T0] = {}
│       &lt;IR for `a = 1;`&gt;
│       // state[T4] = {a: Const{1}}
│  ┌────◄
│  │  }, {
└──│────►
   │    // state[F0] = {}
   │    &lt;IR for `a = 2;`&gt;
   │    // state[F4] = {a: Const{2}}
   ├────◄
   │  })
   └──►
      <b>// state[E0] = {a: <del>Const{1}</del> Unknown}</b>
      ...
</code></pre>

The "join" is arguably the most important part of the dataflow analysis. We will
now discuss it in more detail:

> ### Concepts: Lattice and Join
>
> The types of states, both on program points and on values, are mathematical
> “lattices”. The most important feature of a lattice is that you can `join()`
> two elements.
>
> For example, we already know intuitively that, for a state on a value:
>
> ```
> Join(Const{M}, Const{N}) = Unknown (if M != N)
> ```
>
> This means that if a `%value` might take two different constant values in two
> program paths, then when the program paths merge, we don't know what value it
> takes, hence we have to say it’s `Unknown`.
>
> Here is another example:
>
> ```
> Join(Const{N}, Unknown) = Unknown
> ```
>
> This means that if a `%value` takes a constant literal in one path, but is
> `Unknown` in another path, then when the program paths merge, we have to be on
> the conservative side and say it’s `Unknown`.
>
> What’s less obvious is that, in addition to `Unknown` we also need another
> state value called `Uninit`, and that:
>
> ```
> Join(Uninit,   Const{N}) = Const{N}
> ```
>
> To explain this, let’s go back to the `if`-statement example. In step 4 we
> first propagate `state[T4]` into `state[E0]`, causing `state[E0]` to become
> `{a: Const{1}}`; in step 6 we propagate `state[F4]` into `state[E0]`, causing
> `state[E0]` to become `{a: Unknown}`.
>
> But what was `state[E0]` initially? We need to initialize `state[E0]` with
> some state which, when joined with `{a: Const{1}}`, yields `{a: Const{1}}`.
> This state is:
>
> ```
> {} default = Uninit
> ```
>
> which means: unless explicitly specified, all symbols have the value `Uninit`.

The final result of the analysis, in full detail, is as follows:

```
      ...
      // state[B0] = [default = Unknown] {}
      %cond = jsir.identifier{"cond"}
      // state[%cond] = Unknown
      // state[B1] = [default = Unknown] {}
┌─────◄
│     jshir.if_statement (%cond) ({
├───────►
│       // state[T0] = [default = Unknown] {}
│       %a_ref_true = jsir.identifier_ref{"a"}
│       // state[%a_ref_true] = Uninit
│       // state[T1] = [default = Unknown] {}
│       %1 = jsir.numeric_literal{1}
│       // state[%1] = Const{1}
│       // state[T2] = [default = Unknown] {}
│       %assign_true = jsir.assignment_expression (%a_ref_true, %1)
│       // state[%assign_true] = Const{1}
│       // state[T3] = [default = Unknown] {a: Const{1}}
│       jsir.expression_statement (%assign_true)
│       // state[T4] = [default = Unknown] {a: Const{1}}
│  ┌────◄
│  │  }, {
└──│────►
   │    // state[F0] = [default = Unknown] {}
   │    %a_ref_false = jsir.identifier_ref{"a"}
   │    // state[%a_ref_false] = Uninit
   │    // state[F1] = [default = Unknown] {}
   │    %2 = jsir.numeric_literal{2}
   │    // state[%2] = Const{2}
   │    // state[F2] = [default = Unknown] {}
   │    %assign_false = jsir.assignment_expression (%a_ref_false, %2)
   │    // state[%assign_false] = Const{2}
   │    // state[F3] = [default = Unknown] {a: Const{2}}
   │    jsir.expression_statement (%assign_false)
   │    // state[F4] = [default = Unknown] {a: Const{2}}
   ├────◄
   │  })
   └──►
      ...
```

## Example 3: While loop

The control flow graph (CFG) of a `while`-statement contains a cycle. Even
though the dataflow analysis algorithm runs in the same way, in this example, we
will demonstrate more explicitly:

*   How `Join()` works in practice;

*   The importance of lattice design for the algorithm to terminate.

Consider the following `while`-statement:

```js
a = 1;
while (cond()) {
  a = a + 2;
}
```

Let’s assume that `cond()` returns a nondeterministic `boolean` (i.e. it might
return a different result every time). Therefore, we have to be conservative and
say we don't know what `a` is, both within and after the loop. More
specifically, we expect the final result of the analysis to be as follows:

```js
// {}
a = 1;
// {a: Const{1}}
while (cond()) {
  // NOTE: We don't know how many iterations will be run, so we don't know the
  // value of `a`.

  // {a: Unknown}
  a = a + 2;
  // {a: Unknown}
}
// {a: Unknown}
```

Note that `a` is `Unknown` within the loop body because we are reasoning about
the combination of all iterations.

Now, similar to the previous examples, we convert the code into JSIR:

```
jshir.while_statement ({
  // The `test` region
  // IR for `cond()`:
  %cond_id = jsir.identifier {"cond"}
  %cond = jsir.call_expression (%cond_id)
  jsir.expr_region_end (%cond)
}, {
  // The `body` region
  // IR for `a = a + 2;`:
  %a_ref_body = jsir.identifier_ref{"a"}
  %a = jsir.identifier{"a"}
  %2 = jsir.numeric_literal{2}
  %add = jsir.binary_expression{"+"} (%a, %2)
  %assign_body = jsir.assignment_expression (%a_ref_body, %add)
  jsir.expression_statement (%assign_body)
})
```

Then, we build CFG edges to represent control flow branches:

```
      ...
      %a_ref_before = jsir.identifier_ref{"a"}
      %1 = jsir.numeric_literal{1}
      %assign_before = jsir.assignment_expression (%a_ref_before, %1)
      jsir.expression_statement (%assign_before)
┌─────◄
│     jshir.while_statement ({
├───────►
│       // IR for `cond()`:
│       %cond = ...
│       jsir.expr_region_end (%cond)
│  ┌────◄
│  │  }, {
│  ├────►
│  │    // IR for `a = a + 2;`:
│  │    %assign_body = ...
│  │    jsir.expression_statement (%assign_body)
└──│────◄
   │  })
   └──►
      ...
```

### Step 1: Entering the `test` region for the first time

Similar to the handling of the `if`-statement, we compute the states before the
`while`-loop, i.e. up to `B4`. The state `{a: Const{1}}` is then propagated to
`T0`, the start of the `test` region.

<pre><code>      ...
      <b>// state[B0] = {}</b>
      &lt;IR for `a = 1;`&gt;
      <b>// state[B4] = {a: Const{1}}</b>
┌─────◄
│     jshir.while_statement ({
├───────►
│       <b>// state[T0] = {a: Const{1}}</b>
│       // IR for `cond()`:
│       %cond_id = jsir.identifier {"cond"}
│       %cond = jsir.call_expression (%cond_id)
│       jsir.expr_region_end (%cond)
│  ┌────◄
│  │  }, {
│  ├────►
│  │    &lt;IR for `a = a + 2;`&gt;
└──│────◄
   │  })
   └──►
      ...
</code></pre>

We can see that, as we enter the `test` region for the first time, `a` holds the
value of `1`. Note that this does not match the expected final result
(`Unknown`). We will see how `a` eventually becomes `Unknown` as we progress
through the algorithm.

### Step 2: First evaluation of `test`

Now we will compute the states in the `test` region, which resembles evaluating
the loop condition for the first time. As stated in the assumption before, we
don't know the return value of `cond()`, so we can only assign `Unknown` to
`%cond`.

<pre><code>      ...
      // state[B0] = {}
      &lt;IR for `a = 1;`&gt;
      // state[B4] = {a: Const{1}}
┌─────◄
│     jshir.while_statement ({
├───────►
│       // state[T0] = {a: Const{1}}
│       // IR for `cond()`:
│       %cond_id = jsir.identifier {"cond"}
│       <b>// state[%cond_id] = Unknown</b>
│       <b>// state[T1] = {a: Const{1}}</b>
│       %cond = jsir.call_expression (%cond_id)
│       <b>// state[%cond] = Unknown</b>
│       <b>// state[T2] = {a: Const{1}}</b>
│       jsir.expr_region_end (%cond)
│       <b>// state[T3] = {a: Const{1}}</b>
│  ┌────◄
│  │  }, {
│  ├────►
│  │    &lt;IR for `a = a + 2;`&gt;
└──│────◄
   │  })
   └──►
      ...
</code></pre>

Since `%cond` is `Unknown`, we must conservatively assume that it is both
possible to enter the loop body and exit the loop. Therefore, we propagate
`state[T3]` to both `I0` and `E0`:

<pre><code>      ...
      // state[B0] = {}
      &lt;IR for `a = 1;`&gt;
      // state[B4] = {a: Const{1}}
┌─────◄
│     jshir.while_statement ({
├───────►
│       // state[T0] = {a: Const{1}}
│       %cond = &lt;IR for `cond()`&gt;
│       // state[%cond] = Unknown
│       jsir.expr_region_end (%cond)
│       // state[T3] = {a: Const{1}}
│  ┌────◄
│  │  }, {
│  ├────►
│  │    <b>// state[I0] = {a: Const{1}}</b>
│  │    &lt;IR for `a = a + 2;`&gt;
└──│────◄
   │  })
   └──►
      <b>// state[E0] = {a: Const{1}}</b>
      ...
</code></pre>

> **Important:** Similar to the `if`-statement example, we are now reaching an
> intermediate state where `a` is considered to hold the value `Const{1}`. This
> will be overwritten in later steps.

### Step 3: First evaluation of the `body`

Now, depending on the status of the work queue, the algorithm might decide to
visit ops in the `body` region or after the loop, or both (interleaved). Since
the order doesn't affect the final results, for simplicity reasons, we assume
that we will visit the `body` region. This represents the states during the
first iteration of the loop body, which changes `a` from `1` to `3`.

<pre><code>      ...
      // state[B0] = {}
      &lt;IR for `a = 1;`&gt;
      // state[B4] = {a: Const{1}}
┌─────◄
│     jshir.while_statement ({
├───────►
│       // state[T0] = {a: Const{1}}
│       %cond = &lt;IR for `cond()`&gt;
│       jsir.expr_region_end (%cond)
│       // state[T3] = {a: Const{1}}
│  ┌────◄
│  │  }, {
│  ├────►
│  │    // state[I0] = {a: Const{1}}
│  │    &lt;IR for `a = a + 2;`&gt;
│  │    <b>// state[I6] = {a: Const{3}}</b>
└──│────◄
   │  })
   └──►
      // state[E0] = {a: Const{1}}
      ...
</code></pre>

At the end of the loop body, we jump back to the `test` region, which `Join`s
`state[I6]` into `state[T0]`.

<pre><code>      ...
      // state[B0] = {}
      &lt;IR for `a = 1;`&gt;
      // state[B4] = {a: Const{1}}
┌─────◄
│     jshir.while_statement ({
├───────►
│       <b>// state[T0] = {a: <del>Const{1}</del> Unknown}</b>
│       %cond = &lt;IR for `cond()`&gt;
│       jsir.expr_region_end (%cond)
│       // state[T3] = {a: Const{1}}
│  ┌────◄
│  │  }, {
│  ├────►
│  │    // state[I0] = {a: Const{1}}
│  │    &lt;IR for `a = a + 2;`&gt;
│  │    // state[I6] = {a: Const{3}}
└──│────◄
   │  })
   └──►
      // state[E0] = {a: Const{1}}
      ...
</code></pre>

### Step 4: Re-evaluation of `test` and propagation

Now that `T0` is changed to `{a: Unknown}`, we go through the `test` region
again with this new state, and propagate it to `I0` and `E0`.

<pre><code>      ...
      // state[B0] = {}
      &lt;IR for `a = 1;`&gt;
      // state[B4] = {a: Const{1}}
┌─────◄
│     jshir.while_statement ({
├───────►
│       // state[T0] = {a: Unknown}
│       %cond = &lt;IR for `cond()`&gt;
│       <b>// state[%cond] = Unknown</b>
│       <b>// state[T2] = {a: <del>Const{1}</del> Unknown}</b>
│       jsir.expr_region_end (%cond)
│       <b>// state[T3] = {a: <del>Const{1}</del> Unknown}</b>
│  ┌────◄
│  │  }, {
│  ├────►
│  │    <b>// state[I0] = {a: <del>Const{1}</del> Unknown}</b>
│  │    // IR for `a = a + 2;`:
│  │    ...
│  │    // state[I6] = {a: Const{3}}
└──│────◄
   │  })
   └──►
      <b>// state[E0] = {a: <del>Const{1}</del> Unknown}</b>
      ...
</code></pre>

### Step 5: Re-evaluation of `body` and termination

With `I0` updated to `{a: Unknown}`, we traverse the `body` region again. Since
`a` is `Unknown`, `a = a + 2` results in `a` being `Unknown`. The state at the
end of the `body` region `I6` becomes `{a: Unknown}`.

<pre><code>      ...
      // state[B0] = {}
      &lt;IR for `a = 1;`&gt;
      // state[B4] = {a: Const{1}}
┌─────◄
│     jshir.while_statement ({
├───────►
│       // state[T0] = {a: Unknown}
│       // IR for `cond()`:
│       ...
│       jsir.expr_region_end (%cond)
│       // state[T3] = {a: Unknown}
│  ┌────◄
│  │  }, {
│  ├────►
│  │    // state[I0] = {a: Unknown}
│  │    // IR for `a = a + 2;`:
│  │    ...
│  │    <b>// state[I6] = {a: <del>Const{3}</del> Unknown}</b>
└──│────◄
   │  })
   └──►
      // state[E0] = {a: Unknown}
      ...
</code></pre>

We propagate `I6={a: Unknown}` to `T0`, but `T0` is already `{a: Unknown}`, so
`Join({a: Unknown}, {a: Unknown})` doesn't change `T0`. No new states are
generated, the fixpoint is reached, and the algorithm terminates.

<!--  TODO: Explain "lattice order" in more detail, maybe when introducing the
concept of lattice -->

> ### Concept: Monotonicity
>
> Since the control flow graph contains a cycle now, it is important for the
> dataflow analysis algorithm to avoid getting into an infinite loop.
>
> To achieve this, we just need to design our analysis with the following
> guarantees:
>
> #### Monotonicity
>
> Our analysis must be monotone, meaning the state at each program point can
> only "grow" in one direction. For example, in our constant propagation
> analysis, the state attached to a value can change from `Uninit` to
> `Const{...}`, or from `Const{...}` to `Unknown`, but never in the opposite
> direction.
>
> In practice, this means that the transfer function `f` must "preserve lattice
> order", i.e. when `x ≤ y`, `f(x) ≤ f(y)`.
>
> #### Finite height lattice
>
> Our lattice must have a finite height, i.e. it can only "grow" a finite number
> of times. For example, in our constant propagation analysis, the lattice
> attached to a value can grow at most two times.
