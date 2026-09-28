"""Abstract Syntax Tree (AST) node definitions and interpreter for Task 5."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union


class ASTNode:
    """Base class for all AST nodes."""
    pass


@dataclass(frozen=True)
class Const(ASTNode):
    value: Union[int, float, bool, str]

    def __repr__(self):
        return f"Const({repr(self.value)})"


@dataclass(frozen=True)
class Var(ASTNode):
    name: str

    def __repr__(self):
        return f"Var({self.name})"


@dataclass(frozen=True)
class UnaryOp(ASTNode):
    op: str  # "-", "not"
    operand: ASTNode

    def __repr__(self):
        return f"UnaryOp('{self.op}', {self.operand})"


@dataclass(frozen=True)
class BinaryOp(ASTNode):
    left: ASTNode
    op: str  # "+", "-", "*", "/", "==", "!=", "<", ">", "and", "or"
    right: ASTNode

    def __repr__(self):
        return f"BinaryOp({self.left}, '{self.op}', {self.right})"


@dataclass(frozen=True)
class Assign(ASTNode):
    name: str
    expr: ASTNode

    def __repr__(self):
        return f"Assign('{self.name}', {self.expr})"


@dataclass(frozen=True)
class IfExpr(ASTNode):
    cond: ASTNode
    then_body: ASTNode
    else_body: ASTNode

    def __repr__(self):
        return f"IfExpr({self.cond}, {self.then_body}, {self.else_body})"


@dataclass(frozen=True)
class Block(ASTNode):
    statements: List[ASTNode] = field(default_factory=list)

    def __repr__(self):
        return f"Block({self.statements})"


@dataclass(frozen=True)
class Program(ASTNode):
    statements: List[ASTNode] = field(default_factory=list)

    def __repr__(self):
        return f"Program({self.statements})"


def eval_ast(node: ASTNode, env: Optional[Dict[str, Any]] = None) -> Any:
    """Evaluate an AST node within a variable environment."""
    if env is None:
        env = {}

    if isinstance(node, Const):
        return node.value

    if isinstance(node, Var):
        if node.name not in env:
            raise NameError(f"Undefined variable: {node.name}")
        return env[node.name]

    if isinstance(node, UnaryOp):
        val = eval_ast(node.operand, env)
        if node.op == "-":
            return -val
        elif node.op == "not":
            return not val
        raise ValueError(f"Unknown unary operator: {node.op}")

    if isinstance(node, BinaryOp):
        lval = eval_ast(node.left, env)
        rval = eval_ast(node.right, env)
        if node.op == "+":
            return lval + rval
        elif node.op == "-":
            return lval - rval
        elif node.op == "*":
            return lval * rval
        elif node.op == "/":
            return lval // rval if isinstance(lval, int) and isinstance(rval, int) else lval / rval
        elif node.op == "==":
            return lval == rval
        elif node.op == "!=":
            return lval != rval
        elif node.op == "<":
            return lval < rval
        elif node.op == ">":
            return lval > rval
        elif node.op == "and":
            return lval and rval
        elif node.op == "or":
            return lval or rval
        raise ValueError(f"Unknown binary operator: {node.op}")

    if isinstance(node, Assign):
        val = eval_ast(node.expr, env)
        env[node.name] = val
        return val

    if isinstance(node, IfExpr):
        cond_val = eval_ast(node.cond, env)
        if cond_val:
            return eval_ast(node.then_body, env)
        else:
            return eval_ast(node.else_body, env)

    if isinstance(node, (Block, Program)):
        result = None
        for stmt in node.statements:
            result = eval_ast(stmt, env)
        return result

    raise TypeError(f"Cannot evaluate node of type {type(node).__name__}")
