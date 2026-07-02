// JSIR:      "jsir.file"() <{comments = []}> ({
// JSIR-NEXT:   "jsir.program"() <{source_type = "script"}> ({
// JSIR-NEXT:     %0 = "jsir.template_element_value"() <{cooked = "a", raw = "a"}> : () -> !jsir.any
// JSIR-NEXT:     %1 = "jsir.template_element"(%0) <{tail = false}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %2 = "jsir.template_element_value"() <{cooked = "c", raw = "c"}> : () -> !jsir.any
// JSIR-NEXT:     %3 = "jsir.template_element"(%2) <{tail = false}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %4 = "jsir.template_element_value"() <{cooked = "", raw = ""}> : () -> !jsir.any
// JSIR-NEXT:     %5 = "jsir.template_element"(%4) <{tail = true}> : (!jsir.any) -> !jsir.any
// JSIR-NEXT:     %6 = "jsir.identifier"() <{name = "b"}> : () -> !jsir.any
// JSIR-NEXT:     %7 = "jsir.identifier"() <{name = "d"}> : () -> !jsir.any
// JSIR-NEXT:     %8 = "jsir.template_literal"(%1, %3, %5, %6, %7) <{operandSegmentSizes = array<i32: 3, 2>}> : (!jsir.any, !jsir.any, !jsir.any, !jsir.any, !jsir.any) -> !jsir.any
// JSIR-NEXT:     "jsir.expression_statement"(%8) : (!jsir.any) -> ()
// JSIR-NEXT:   }, {
// JSIR-NEXT:   ^bb0:
// JSIR-NEXT:   }) : () -> ()
// JSIR-NEXT: }) : () -> ()
