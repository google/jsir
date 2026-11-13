function sideEffect() {
  return 1;
}
class Foo {
  method(a, b = sideEffect()) {
    return a + b;
  }
  static staticMethod(c, d = sideEffect()) {
    return c + d;
  }
  set value(e = sideEffect()) {
    this._value = e;
  }
  #privateMethod(f, g = sideEffect()) {
    return f + g;
  }
  [sideEffect()](h, i = sideEffect()) {
    return h + i;
  }
}
