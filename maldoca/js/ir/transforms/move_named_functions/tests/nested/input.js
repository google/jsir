function outer_function_1() {
  1;
  function inner_function() {}
  2;
  {
    3;
    function function_in_block() {}
  }
}

function outer_function_2() {
  if (x)
    return;
  foo();
  function foo() {}
}

function outer_function_3() {
  function foo() {}
  if (x)
    return;
  foo();
  function bar() {}
}

function outer_function_4() {
  function foo() {}
}

function outer_function_5() {
  f();
  {
    function bar() {}
  }
}

function outer_function_6() {
  f();
  if (true) {
    function bar() {}
  }
}
