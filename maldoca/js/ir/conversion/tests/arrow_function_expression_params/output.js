// SOURCE:      function sideEffect() {
// SOURCE-NEXT:   return 1;
// SOURCE-NEXT: }
// SOURCE-NEXT: const f = (a, b = sideEffect(), ...rest) => a + b;
