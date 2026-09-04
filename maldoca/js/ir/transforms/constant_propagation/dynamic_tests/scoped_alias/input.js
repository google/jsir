// exec:begin
function prelude_1() {
  return 'prelude_1';
}
function prelude_2() {
  return 'prelude_2';
}
function prelude_3() {
  return 'prelude_3';
}
// exec:end

console.log(prelude_1());

let wrapper = prelude_1;
console.log(wrapper());

{
  let wrapper = prelude_2;
  console.log(wrapper());
  {
    console.log(wrapper());
  }
}

function foo() {
  let wrapper = prelude_3;
  console.log(wrapper());
}
