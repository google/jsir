function sideEffect() {
  return 1;
}
const f = (a, b = sideEffect(), ...rest) => a + b;
