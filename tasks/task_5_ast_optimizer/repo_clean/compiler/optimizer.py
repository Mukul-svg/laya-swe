"""Multi-Pass AST Optimizer: Constant Folding & Dead Code Elimination."""
from typing import Set, List
from compiler.ast_nodes import (
    ASTNode, Const, Var, UnaryOp, BinaryOp, Assign, IfExpr, Block, Program
)


class ASTOptimizer:
    """Optimizes expression and statement ASTs."""

    def fold_constants(self, node: ASTNode) -> ASTNode:
        """Pass 1: Recursively fold constant subexpressions and algebraic identities.
        
        Rules to support:
        1. Binary operations between two Const nodes: fold to single Const.
        2. Unary operations on Const:
           - UnaryOp('-', Const(v)) -> Const(-v)
           - UnaryOp('not', Const(v)) -> Const(not v)
        3. Algebraic identities with Var or complex subexpressions:
           - x + 0 or 0 + x -> x
           - x - 0 -> x
           - x * 1 or 1 * x -> x
           - x * 0 or 0 * x -> Const(0)
           - True and x -> x
           - False and x -> Const(False)
           - False or x -> x
           - True or x -> Const(True)
        """
        if isinstance(node, (Const, Var)):
            return node

        if isinstance(node, UnaryOp):
            operand = self.fold_constants(node.operand)
            # BUG: UnaryOp constant folding is unimplemented in clean stub!
            return UnaryOp(node.op, operand)

        if isinstance(node, BinaryOp):
            left = self.fold_constants(node.left)
            right = self.fold_constants(node.right)

            # 1. Both operands are Const
            if isinstance(left, Const) and isinstance(right, Const):
                lv, rv = left.value, right.value
                try:
                    if node.op == "+":
                        return Const(lv + rv)
                    elif node.op == "-":
                        return Const(lv - rv)
                    elif node.op == "*":
                        return Const(lv * rv)
                    elif node.op == "/":
                        return Const(lv // rv if isinstance(lv, int) and isinstance(rv, int) else lv / rv)
                    elif node.op == "==":
                        return Const(lv == rv)
                    elif node.op == "!=":
                        return Const(lv != rv)
                    elif node.op == "<":
                        return Const(lv < rv)
                    elif node.op == ">":
                        return Const(lv > rv)
                    elif node.op == "<=":
                        return Const(lv <= rv)
                    elif node.op == ">=":
                        return Const(lv >= rv)
                    elif node.op == "and":
                        return Const(bool(lv and rv))
                    elif node.op == "or":
                        return Const(bool(lv or rv))
                except Exception:
                    pass

            # BUG: Algebraic identities (x * 0, x + 0, x * 1, etc.) are unimplemented in clean stub!
            return BinaryOp(left, node.op, right)

        if isinstance(node, Assign):
            return Assign(node.name, self.fold_constants(node.expr))

        if isinstance(node, IfExpr):
            cond = self.fold_constants(node.cond)
            then_b = self.fold_constants(node.then_body)
            else_b = self.fold_constants(node.else_body)
            return IfExpr(cond, then_b, else_b)

        if isinstance(node, Block):
            return Block([self.fold_constants(s) for s in node.statements])

        if isinstance(node, Program):
            return Program([self.fold_constants(s) for s in node.statements])

        return node

    def get_referenced_vars(self, node: ASTNode) -> Set[str]:
        """Collect all variable names read by this AST subtree."""
        refs: Set[str] = set()
        if isinstance(node, Var):
            refs.add(node.name)
        elif isinstance(node, UnaryOp):
            refs.update(self.get_referenced_vars(node.operand))
        elif isinstance(node, BinaryOp):
            refs.update(self.get_referenced_vars(node.left))
            refs.update(self.get_referenced_vars(node.right))
        elif isinstance(node, Assign):
            refs.update(self.get_referenced_vars(node.expr))
        elif isinstance(node, IfExpr):
            refs.update(self.get_referenced_vars(node.cond))
            refs.update(self.get_referenced_vars(node.then_body))
            refs.update(self.get_referenced_vars(node.else_body))
        elif isinstance(node, (Block, Program)):
            for s in node.statements:
                refs.update(self.get_referenced_vars(s))
        return refs

    def eliminate_dead_code(self, node: ASTNode) -> ASTNode:
        """Pass 2: Eliminate dead branches and dead store assignments.
        
        Rules to support:
        1. IfExpr with constant condition:
           - IfExpr(Const(True), then_b, else_b) -> then_b
           - IfExpr(Const(False), then_b, else_b) -> else_b
        2. Dead Store Elimination in Block:
           - In a sequence of statements in a Block, if Assign(var, expr) occurs,
             but `var` is NOT in the referenced variables of any subsequent statements
             AND it is not the final expression of the block, eliminate the Assign!
        """
        # BUG: Dead code elimination is completely unimplemented in clean stub!
        return node

    def optimize(self, node: ASTNode, max_passes: int = 10) -> ASTNode:
        """Run iterative optimization passes until fixed point."""
        current = node
        for _ in range(max_passes):
            folded = self.fold_constants(current)
            dead_eliminated = self.eliminate_dead_code(folded)
            if dead_eliminated == current:
                break
            current = dead_eliminated
        return current
