// NOTE:
//
// In non-strict mode, JavaScript allows a function to be defined inside an if
// statement, even if the body is not wrapped in a block.
//
// In this case, the function declaration should NOT be moved out of the if
// statement. This is because whether the function is actually declared depends
// on condition of the if statement.
//
// +-----------------+-----------+----------------+------------------------+
// |                 | top-level | inside a block | inside an if statement |
// +-----------------+-----------+----------------+------------------------+
// |   strict mode   |     ✓     |       ✓        |           ✗            |
// | non-strict mode |     ✓     |       ✓        |           ✓            |
// +-----------------+-----------+----------------+------------------------+

function before_if() {}
if (x)
  function in_if() {}
function after_if() {}
