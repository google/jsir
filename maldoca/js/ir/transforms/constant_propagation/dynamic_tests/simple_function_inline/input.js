function add(a, b) {
  return a + b;
}

function add_multiline(a, b) {
  let c = a + b;
  return c;
}

function add_buggy(a, b) {
  return;
}

console.log(add(1, 2));
console.log(add_multiline(1, 2));
console.log(add_buggy(1, 2));
