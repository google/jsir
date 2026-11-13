// SOURCE:      function sideEffect() {
// SOURCE-NEXT:   return 1;
// SOURCE-NEXT: }
// SOURCE-NEXT: const obj = {
// SOURCE-NEXT:   method(a, b = sideEffect()) {
// SOURCE-NEXT:     return a + b;
// SOURCE-NEXT:   },
// SOURCE-NEXT:   [sideEffect()](c, d = sideEffect()) {
// SOURCE-NEXT:     return c + d;
// SOURCE-NEXT:   },
// SOURCE-NEXT:   get prop() {
// SOURCE-NEXT:     return 1;
// SOURCE-NEXT:   }
// SOURCE-NEXT: };
