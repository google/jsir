// TODO(b/217662731): `is_a_const` should be a const.

var not_a_const = 1;
while (x) {
  not_a_const = not_a_const + 1;
  var is_a_const = 1;
}
console.log(not_a_const);
console.log(is_a_const);
