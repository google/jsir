// TODO(b/330592752): Fix hoisting so that we identify `is_a_const` as a
// constant.

var not_a_const = 1;
while (x) {
  not_a_const = not_a_const + 1;
  var is_a_const = 1;
}
console.log(not_a_const);
console.log(is_a_const);
