let a = 0;

if (true) {
  let b = 1;
} else {
  let dead_1 = 0;
}

if (true) {
  if (false) {
    let nested_dead_2 = 1;
  } else {
    let c = 2;
  }
  let d = 3;
}

if (false) {
  let dead_3 = 0;
} else {
  let e = 4;
}

let f = 5;

if (true)
  f = f + 1;
else
  var dead_7 = 0;

if (false)
  var dead_8 = 0;
else
  f = f + 1;

if (true) {
  let g = 6;
}

if (false) {
  let dead_4 = 0;
}

if (true)
  var h = 7;
else
  var dead_5 = 0;

if (false)
  var dead_6 = 0;
else
  var i = 8;
