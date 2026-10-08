// exec:begin
function prelude(x) {
  return x;
}
// exec:end

console.log(prelude('raw'));

let alias = prelude;
console.log(alias('alias'));

function wrapper(x) {
  return prelude(x);
}
console.log(wrapper('wrapper'));

let wrapper_alias = wrapper;
console.log(wrapper_alias('wrapper_alias'));

function alias_wrapper(x) {
  return alias(x);
}
console.log(alias_wrapper('alias_wrapper'));
