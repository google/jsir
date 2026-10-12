function foo(x) {
  return x + 1;
}

function bar(x) {
  return foo((x + 1) * 2);
}

function baz(f, x) {
  return f(x);
}

console.log(foo(1));
console.log(bar(2));
console.log(baz(bar, 2));
