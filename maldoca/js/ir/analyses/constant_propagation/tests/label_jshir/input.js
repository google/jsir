var x = 1;
my_label: while (true) {
  x = 2;
  while (true) {
    break my_label;
  }
  x = 3;
}
console.log(x);
