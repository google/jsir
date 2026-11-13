// SOURCE:      function sideEffect() {
// SOURCE-NEXT:   return 1;
// SOURCE-NEXT: }
// SOURCE-NEXT: class Foo {
// SOURCE-NEXT:   method(a, b = sideEffect()) {
// SOURCE-NEXT:     return a + b;
// SOURCE-NEXT:   }
// SOURCE-NEXT:   static staticMethod(c, d = sideEffect()) {
// SOURCE-NEXT:     return c + d;
// SOURCE-NEXT:   }
// SOURCE-NEXT:   set value(e = sideEffect()) {
// SOURCE-NEXT:     this._value = e;
// SOURCE-NEXT:   }
// SOURCE-NEXT:   #privateMethod(f, g = sideEffect()) {
// SOURCE-NEXT:     return f + g;
// SOURCE-NEXT:   }
// SOURCE-NEXT:   [sideEffect()](h, i = sideEffect()) {
// SOURCE-NEXT:     return h + i;
// SOURCE-NEXT:   }
// SOURCE-NEXT: }
