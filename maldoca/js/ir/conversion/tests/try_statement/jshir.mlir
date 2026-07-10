// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     "jshir.try_statement"() ({
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "error"}> : () -> !jsir.any
// JSIR-NEXT:       "jshir.catch_clause"(%0) ({
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:           %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.try_statement"() ({
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:     "jshir.try_statement"() ({
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       %0 = "jsir.identifier_ref"() <{name = "error"}> : () -> !jsir.any
// JSIR-NEXT:       "jshir.catch_clause"(%0) ({
// JSIR-NEXT:         "jshir.block_statement"() ({
// JSIR-NEXT:           %1 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:           "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSIR-NEXT:         }, {
// JSIR-NEXT:         ^bb0:
// JSIR-NEXT:         }) : () -> ()
// JSIR-NEXT:       }) : (!jsir.any) -> ()
// JSIR-NEXT:     }, {
// JSIR-NEXT:       "jshir.block_statement"() ({
// JSIR-NEXT:         %0 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSIR-NEXT:         "jsir.expression_statement"(%0) : (!jsir.any) -> ()
// JSIR-NEXT:       }, {
// JSIR-NEXT:       ^bb0:
// JSIR-NEXT:       }) : () -> ()
// JSIR-NEXT:     }) : () -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
