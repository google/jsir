// JSHIR:      "jsir.file"() <{comments = []}> ({
// JSHIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSHIR-NEXT:     "jshir.try_statement"() ({
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "error"}> : () -> !jsir.any
// JSHIR-NEXT:       "jshir.catch_clause"(%0) ({
// JSHIR-NEXT:         "jshir.block_statement"() ({
// JSHIR-NEXT:           %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:           "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:         }, {
// JSHIR-NEXT:         ^bb0:
// JSHIR-NEXT:         }) : () -> ()
// JSHIR-NEXT:       }) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jshir.try_statement"() ({
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:     "jshir.try_statement"() ({
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "error"}> : () -> !jsir.any
// JSHIR-NEXT:       "jshir.catch_clause"(%0) ({
// JSHIR-NEXT:         "jshir.block_statement"() ({
// JSHIR-NEXT:           %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSHIR-NEXT:           "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSHIR-NEXT:         }, {
// JSHIR-NEXT:         ^bb0:
// JSHIR-NEXT:         }) : () -> ()
// JSHIR-NEXT:       }) : (!jsir.any) -> ()
// JSHIR-NEXT:     }, {
// JSHIR-NEXT:       "jshir.block_statement"() ({
// JSHIR-NEXT:         %0 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSHIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSHIR-NEXT:       }, {
// JSHIR-NEXT:       ^bb0:
// JSHIR-NEXT:       }) : () -> ()
// JSHIR-NEXT:     }) : () -> ()
// JSHIR-NEXT:   }, {
// JSHIR-NEXT:   ^bb0:
// JSHIR-NEXT:   }) : () -> ()
// JSHIR-NEXT: }) : () -> ()
