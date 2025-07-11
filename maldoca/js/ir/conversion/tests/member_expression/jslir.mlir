// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.member_expression"(%0) <{literal_property = #jsir<identifier   <L 1 C 2>, <L 1 C 3>, "b", 2, 3, 0, "b">}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%1) : (!jsir.any) -> ()
// JSLIR-NEXT:     %2 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %4 = "jsir.member_expression"(%2, %3) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%4) : (!jsir.any) -> ()
// JSLIR-NEXT:     %5 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %6 = "jsir.member_expression_ref"(%5) <{literal_property = #jsir<identifier   <L 5 C 2>, <L 5 C 3>, "b", 15, 16, 0, "b">}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %7 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     %8 = "jsir.assignment_expression"(%6, %7) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%8) : (!jsir.any) -> ()
// JSLIR-NEXT:     %9 = "jsir.identifier"() <{name = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %10 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %11 = "jsir.member_expression_ref"(%9, %10) : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     %12 = "jsir.identifier"() <{name = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     %13 = "jsir.assignment_expression"(%11, %12) <{operator_ = "="}> : (!jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%13) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
