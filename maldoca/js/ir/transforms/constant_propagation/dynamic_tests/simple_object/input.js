let object = {
  'a': 'a',
  'b': '' + 'b',
  c: 'c',
  ['d']: 'd',
};

console.log(object['a']);
console.log(object['' + 'a']);
console.log(object['b']);
console.log(object['' + 'b']);
console.log(object.c);
console.log(object.d);
