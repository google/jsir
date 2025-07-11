// SOURCE:      // =============================================================================
// SOURCE-NEXT: // Breaking out of a while loop
// SOURCE-NEXT: // =============================================================================
// SOURCE-EMPTY:
// SOURCE-NEXT: while (a) {
// SOURCE-NEXT:   b;
// SOURCE-NEXT:   if (c) break;
// SOURCE-NEXT: }
// SOURCE-EMPTY:
// SOURCE-NEXT: // =============================================================================
// SOURCE-NEXT: // Breaking out of second while loop
// SOURCE-NEXT: // =============================================================================
// SOURCE-EMPTY:
// SOURCE-NEXT: label0: while (a) {
// SOURCE-NEXT:   b;
// SOURCE-NEXT:   label1: while (d) if (c) break label0;
// SOURCE-NEXT: }
// SOURCE-EMPTY:
// SOURCE-NEXT: // =============================================================================
// SOURCE-NEXT: // Breaking immediately after label
// SOURCE-NEXT: // =============================================================================
// SOURCE-EMPTY:
// SOURCE-NEXT: label: break label;
