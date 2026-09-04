// exec:begin
function prelude(x) {
  return x;
}
// exec:end

console.log(prelude(0));

function wrapper_1(x) {
  return prelude(x);
}
console.log(wrapper_1(1));

function wrapper_2(x) {
  return wrapper_1(x);
}
console.log(wrapper_2(2));

function wrapper_3(x) {
  return prelude(x + 100);
}
console.log(wrapper_3(3));
