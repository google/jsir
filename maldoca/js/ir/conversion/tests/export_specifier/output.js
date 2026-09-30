// SOURCE:      let x = 1;
// SOURCE-NEXT: let y = 2;
// SOURCE-NEXT: export { x };
// SOURCE-NEXT: export { x as a, y as b };
// SOURCE-NEXT: export { x as "a-b" };
// SOURCE-NEXT: export { "a-b" as c } from "foo";
// SOURCE-NEXT: export { default as d } from "foo";
// SOURCE-NEXT: export var e = 1;
