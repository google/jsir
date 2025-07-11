// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     "jshir.while_statement"() ({
// JSHIR-NEXT:       %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:       "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:         "jshir.if_statement"(%0) ({
// JSHIR-NEXT:           "jshir.continue_statement"() : () -> ()
// JSHIR-NEXT:         }, {
// JSHIR-NEXT:         }) : (!jsir.any) -> ()
// JSHIR-NEXT:         %1 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jshir.labeled_statement"() <{label = #jsir<identifier   <L 7 C 0>, <L 7 C 6>, "label0", 43, 49, 0, "label0">}> ({
// JSHIR-NEXT:       "jshir.while_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expr_region_end"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:         "jshir.block_statement"() ({
// JSHIR-NEXT:           %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:           "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:           "jshir.labeled_statement"() <{label = #jsir<identifier   <L 9 C 2>, <L 9 C 8>, "label1", 70, 76, 4, "label1">}> ({
// JSHIR-NEXT:             "jshir.while_statement"() ({
// JSHIR-NEXT:               %1 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSHIR-NEXT:               "jsir.expr_region_end"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:             }, {
// JSHIR-NEXT:               %1 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:               "jshir.if_statement"(%1) ({
// JSHIR-NEXT:                 "jshir.continue_statement"() <{label = #jsir<identifier   <L 11 C 15>, <L 11 C 21>, "label0", 114, 120, 5, "label0">}> : () -> ()
// JSHIR-NEXT:               }, {
// JSHIR-NEXT:               }) : (!jsir.any) -> ()
// JSHIR-NEXT:             }) : () -> ()
// JSHIR-NEXT:           }) : () -> ()
// JSHIR-NEXT:         }, {
// JSHIR-NEXT:         ^bb0:
// JSHIR-NEXT:         }) : () -> ()
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
