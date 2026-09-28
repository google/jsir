function sideEffect() {
  return 1;
}
const obj = {
  method(a, b = sideEffect()) {
    return a + b;
  },
  [sideEffect()](c, d = sideEffect()) {
    return c + d;
  },
  get prop() {
    return 1;
  }
};
