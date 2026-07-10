// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jshir.while_statement"() ({
// JSIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:         "jshir.if_statement"(%0) ({
// JSIR-NEXT:           "jshir.continue_statement"() : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:         }) : (!jsir.any) -> ()
// JSIR-NEXT:         %1 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.labeled_statement"() <{label = #jsir<identifier <L 7 C 0>, <L 7 C 6>, "label0", 43, 49, 0, "label0">}> ({
// JSIR-NEXT:       "jshir.while_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:           %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:           "jshir.labeled_statement"() <{label = #jsir<identifier <L 9 C 2>, <L 9 C 8>, "label1", 70, 76, 4, "label1">}> ({
// JSIR-NEXT:             "jshir.while_statement"() ({
// JSIR-NEXT:               %1 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSIR-NEXT:               "jsir.expr_region_end"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:             }, {
// JSIR-NEXT:               %1 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:               "jshir.if_statement"(%1) ({
// JSIR-NEXT:                 "jshir.continue_statement"() <{label = #jsir<identifier <L 11 C 15>, <L 11 C 21>, "label0", 114, 120, 5, "label0">}> : () -> ()
// JSIR-NEXT:               }, {
// JSIR-NEXT:               }) : (!jsir.any) -> ()
// JSIR-NEXT:             }) : () -> ()
// JSIR-NEXT:           }) : () -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
