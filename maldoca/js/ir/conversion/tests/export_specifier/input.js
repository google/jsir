let x = 1;
let y = 2;
export {x};
export {x as a, y as b};
export {x as "a-b"};
export {"a-b" as c} from "foo";
export {default as d} from "foo";
export var e = 1;
