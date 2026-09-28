# Task 5: Multi-Pass AST Optimizer (Constant Folding & Dead Code Elimination)

### Objective
Complete the multi-pass AST optimizer in `compiler/optimizer.py` to optimize expression and statement ASTs.

### Steps to Follow
1. Inspect `compiler/ast_nodes.py`, `compiler/optimizer.py`, and `tests/test_optimizer.py`.
2. Run `pytest tests/test_optimizer.py` to see the failing tests.
3. In `compiler/optimizer.py`:
   - Implement `UnaryOp` constant folding in `fold_constants()`:
     - `UnaryOp('-', Const(v))` -> `Const(-v)`
     - `UnaryOp('not', Const(v))` -> `Const(not v)`
   - Implement algebraic & boolean identities in `fold_constants()`:
     - `x * 0` or `0 * x` -> `Const(0)`
     - `x + 0` or `0 + x` -> `x`
     - `x * 1` or `1 * x` -> `x`
     - `x - 0` -> `x`
     - `True and x` -> `x`, `False and x` -> `Const(False)`
     - `False or x` -> `x`, `True or x` -> `Const(True)`
   - Implement `eliminate_dead_code()`:
     - For `IfExpr(cond, then_body, else_body)`: if `cond` is `Const(True)`, return `then_body`; if `cond` is `Const(False)`, return `else_body`.
     - For `Block(statements)`: eliminate dead store assignments. If `Assign(name, expr)` is present, but `name` is NOT referenced in subsequent statements (`self.get_referenced_vars()`) and is not the final statement in the block, drop that assignment from the block!
4. Run `pytest tests/test_optimizer.py` to verify all 7 tests pass.
