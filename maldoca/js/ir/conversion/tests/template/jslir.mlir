// JSLIR:      "jsir.file"() <{comments = []}> ({
// JSLIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSLIR-NEXT:     %0 = "jsir.template_element_value"() <{cooked = "a", raw = "a"}> : () -> !jsir.any
// JSLIR-NEXT:     %1 = "jsir.template_element"(%0) <{tail = false}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %2 = "jsir.template_element_value"() <{cooked = "c", raw = "c"}> : () -> !jsir.any
// JSLIR-NEXT:     %3 = "jsir.template_element"(%2) <{tail = false}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %4 = "jsir.template_element_value"() <{cooked = "", raw = ""}> : () -> !jsir.any
// JSLIR-NEXT:     %5 = "jsir.template_element"(%4) <{tail = true}> : (!jsir.any) -> !jsir.any
// JSLIR-NEXT:     %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSLIR-NEXT:     %7 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSLIR-NEXT:     %8 = "jsir.template_literal"(%1, %3, %5, %6, %7) <{operandSegmentSizes = array<i32: 3, 2>}> : (!jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any) -> !jsir.any
// JSLIR-NEXT:     "jsir.expression_statement"(%8) : (!jsir.any) -> ()
// JSLIR-NEXT:   }, {
// JSLIR-NEXT:   ^bb0:
// JSLIR-NEXT:   }) : () -> ()
// JSLIR-NEXT: }) : () -> ()
