var x = 1;
try {
  x = 2;
} catch (e) {
  x = 3;
} finally {
  x = 4;
}
console.log(x);
