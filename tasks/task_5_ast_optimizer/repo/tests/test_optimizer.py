"""Ground truth unit tests for Task 5: AST Optimizer."""
import os
import sys
import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(cur_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from compiler.ast_nodes import (
    Const, Var, UnaryOp, BinaryOp, Assign, IfExpr, Block, eval_ast
)
from compiler.optimizer import ASTOptimizer


def test_basic_constant_folding():
    """Verify basic binary arithmetic folding."""
    opt = ASTOptimizer()
    expr = BinaryOp(Const(2), "+", BinaryOp(Const(3), "*", Const(4)))
    res = opt.optimize(expr)
    assert isinstance(res, Const)
    assert res.value == 14


def test_unary_constant_folding():
    """Verify unary operations are folded on constant operands."""
    opt = ASTOptimizer()
    expr1 = UnaryOp("-", Const(10))
    res1 = opt.optimize(expr1)
    assert isinstance(res1, Const)
    assert res1.value == -10

    expr2 = UnaryOp("not", Const(False))
    res2 = opt.optimize(expr2)
    assert isinstance(res2, Const)
    assert res2.value is True

    expr3 = UnaryOp("-", UnaryOp("-", Const(42)))
    res3 = opt.optimize(expr3)
    assert isinstance(res3, Const)
    assert res3.value == 42


def test_algebraic_identities():
    """Verify algebraic simplifications for multiplication, addition, and subtraction."""
    opt = ASTOptimizer()

    # x * 0 -> 0 and 0 * x -> 0
    e1 = BinaryOp(Var("x"), "*", Const(0))
    assert opt.optimize(e1) == Const(0)

    e2 = BinaryOp(Const(0), "*", Var("x"))
    assert opt.optimize(e2) == Const(0)

    # x + 0 -> x and 0 + x -> x
    e3 = BinaryOp(Var("y"), "+", Const(0))
    assert opt.optimize(e3) == Var("y")

    e4 = BinaryOp(Const(0), "+", Var("y"))
    assert opt.optimize(e4) == Var("y")

    # x * 1 -> x and 1 * x -> x
    e5 = BinaryOp(Var("z"), "*", Const(1))
    assert opt.optimize(e5) == Var("z")

    e6 = BinaryOp(Const(1), "*", Var("z"))
    assert opt.optimize(e6) == Var("z")

    # x - 0 -> x
    e7 = BinaryOp(Var("w"), "-", Const(0))
    assert opt.optimize(e7) == Var("w")


def test_boolean_identities():
    """Verify boolean logic short-circuiting and identities."""
    opt = ASTOptimizer()

    # True and x -> x
    e1 = BinaryOp(Const(True), "and", Var("p"))
    assert opt.optimize(e1) == Var("p")

    # False and x -> False
    e2 = BinaryOp(Const(False), "and", Var("p"))
    assert opt.optimize(e2) == Const(False)

    # False or x -> x
    e3 = BinaryOp(Const(False), "or", Var("q"))
    assert opt.optimize(e3) == Var("q")

    # True or x -> True
    e4 = BinaryOp(Const(True), "or", Var("q"))
    assert opt.optimize(e4) == Const(True)


def test_dead_if_branch_elimination():
    """Verify dead condition branches are pruned."""
    opt = ASTOptimizer()

    # IfExpr(True, active, dead) -> active
    e1 = IfExpr(Const(True), Var("active_branch"), Var("dead_branch"))
    assert opt.optimize(e1) == Var("active_branch")

    # IfExpr(False, dead, active) -> active
    e2 = IfExpr(Const(False), Var("dead_branch"), Var("active_branch"))
    assert opt.optimize(e2) == Var("active_branch")

    # IfExpr with constant condition evaluated by folding: (2 > 1) -> True -> active
    e3 = IfExpr(BinaryOp(Const(2), ">", Const(1)), Var("branch_a"), Var("branch_b"))
    assert opt.optimize(e3) == Var("branch_a")


def test_dead_store_elimination():
    """Verify assignments to variables never subsequently read are removed."""
    opt = ASTOptimizer()

    # Statement 1: unused_var = 100 (never read)
    # Statement 2: x = 10
    # Statement 3: x + 5 (returns 15)
    blk = Block([
        Assign("unused_var", Const(100)),
        Assign("x", Const(10)),
        BinaryOp(Var("x"), "+", Const(5))
    ])

    optimized_blk = opt.optimize(blk)
    assert isinstance(optimized_blk, Block)

    # unused_var assignment MUST be eliminated!
    assigned_names = [s.name for s in optimized_blk.statements if isinstance(s, Assign)]
    assert "unused_var" not in assigned_names
    assert "x" in assigned_names

    # Semantics must be preserved
    assert eval_ast(optimized_blk) == eval_ast(blk)


def test_full_optimization_pipeline():
    """Verify full multi-pass optimization on a compound program block."""
    opt = ASTOptimizer()

    # Compound AST:
    # a = 5 * 0          (folds to a = 0)
    # dead_flag = not True (dead store, folds to False then eliminated)
    # y = 10 + 20        (folds to y = 30)
    # if (y > 25): y * 2 else: 0  (folds condition to True, then body to 30 * 2 -> 60)
    program = Block([
        Assign("a", BinaryOp(Const(5), "*", Const(0))),
        Assign("dead_temp", UnaryOp("not", Const(True))),
        Assign("y", BinaryOp(Const(10), "+", Const(20))),
        IfExpr(
            BinaryOp(Var("y"), ">", Const(25)),
            BinaryOp(Var("y"), "*", Const(2)),
            Const(0)
        )
    ])

    optimized = opt.optimize(program)
    assert isinstance(optimized, Block)

    # dead_temp should be pruned
    assigned_names = [s.name for s in optimized.statements if isinstance(s, Assign)]
    assert "dead_temp" not in assigned_names

    # Resulting value evaluation should match
    env_orig = {}
    orig_val = eval_ast(program, env_orig)
    env_opt = {}
    opt_val = eval_ast(optimized, env_opt)
    assert orig_val == opt_val == 60
