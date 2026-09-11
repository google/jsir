// exec:begin
function prelude(x) {
  return x;
}
// exec:end

console.log(prelude(0));

let wrapper_1 = prelude;
console.log(wrapper_1(1));

var wrapper_2 = wrapper_1;
console.log(wrapper_2(2));

const wrapper_3 = wrapper_2;
console.log(wrapper_3(3));
