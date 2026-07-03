Currently maldoca/astgen is written in C++. Let's rewrite it in python.

Note:

* First make sure that py_{binary,library,test} works in this repo.

* All tests must be ported and still pass. In particular:

  * maldoca/astgen/test contains several golden test files
  
  * maldoca/js contains a lot of generated files

* Please do this piece by piece, in whatever order you think is reasonable.

  Whenever you ported one component, write or port the corresponding tests.
