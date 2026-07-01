// JSIR:      "jsir.file"() <{comments = [#jsir<comment_line <L 1 C 0>, <L 1 C 80>, 0, 80, " =============================================================================">, #jsir<comment_line <L 2 C 0>, <L 2 C 31>, 81, 112, " Breaking out of a while loop">, #jsir<comment_line <L 3 C 0>, <L 3 C 80>, 113, 193, " =============================================================================">, #jsir<comment_line <L 11 C 0>, <L 11 C 80>, 235, 315, " =============================================================================">, #jsir<comment_line <L 12 C 0>, <L 12 C 36>, 316, 352, " Breaking out of second while loop">, #jsir<comment_line <L 13 C 0>, <L 13 C 80>, 353, 433, " =============================================================================">, #jsir<comment_line <L 22 C 0>, <L 22 C 80>, 514, 594, " =============================================================================">, #jsir<comment_line <L 23 C 0>, <L 23 C 35>, 595, 630, " Breaking immediately after label">, #jsir<comment_line <L 24 C 0>, <L 24 C 80>, 631, 711, " =============================================================================">]}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jshir.while_statement"() ({
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:         %1 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:         "jshir.if_statement"(%1) ({
// JSIR-NEXT:           "jshir.break_statement"() : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:         }) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.labeled_statement"() <{label = #jsir<identifier <L 15 C 0>, <L 15 C 6>, "label0", 435, 441, 0, "label0">}> ({
// JSIR-NEXT:       "jshir.while_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:           %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:           "jshir.labeled_statement"() <{label = #jsir<identifier <L 17 C 2>, <L 17 C 8>, "label1", 462, 468, 4, "label1">}> ({
// JSIR-NEXT:             "jshir.while_statement"() ({
// JSIR-NEXT:               %1 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSIR-NEXT:               "jsir.expr_region_end"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:             }, {
// JSIR-NEXT:               %1 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:               "jshir.if_statement"(%1) ({
// JSIR-NEXT:                 "jshir.break_statement"() <{label = #jsir<identifier <L 19 C 12>, <L 19 C 18>, "label0", 503, 509, 5, "label0">}> : () -> ()
// JSIR-NEXT:               }, {
// JSIR-NEXT:               }) : (!jsir.any) -> ()
// JSIR-NEXT:             }) : () -> ()
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.labeled_statement"() <{label = #jsir<identifier <L 26 C 0>, <L 26 C 5>, "label", 713, 718, 0, "label">}> ({
// JSIR-NEXT:       "jshir.break_statement"() <{label = #jsir<identifier <L 26 C 13>, <L 26 C 18>, "label", 726, 731, 0, "label">}> : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
