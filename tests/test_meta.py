# SPDX-License-Identifier: MPL-2.0
"""Meta-tests: do the stubs match the runtime, and do the other tests exercise all of them?"""

from __future__ import annotations

import ast
import importlib
import inspect
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from inspect import Parameter as P
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).parent
SRC = HERE.parent / "src"


@dataclass
class Member:
    module: str
    cls: str | None
    name: str
    defs: tuple[ast.FunctionDef, ...]  # empty for classes, attributes and properties

    @property
    def id(self) -> str:
        return ".".join(filter(None, [self.module, self.cls, self.name]))

    def params(self) -> Generator[list[P]]:
        """Parameters of each (overload) definition, without `self`."""
        for f in self.defs:
            params = _stub_params(f)
            yield params[1:] if self.cls else params


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    parts[0] = parts[0].removesuffix("-stubs")
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _is_property(f: ast.FunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "property" for d in f.decorator_list)


def _members(body: Iterable[ast.stmt], module: str, cls: str | None = None) -> Iterator[Member]:
    defs: defaultdict[str, list[ast.FunctionDef]] = defaultdict(list)
    for node in body:
        match node:
            case ast.FunctionDef(name=name) if _is_property(node):
                yield Member(module, cls, name, ())
            case ast.FunctionDef(name=name):
                defs[name].append(node)
            case ast.AnnAssign(target=ast.Name(id=name)):
                yield Member(module, cls, name, ())
            case ast.ClassDef(name=name) if cls is None:
                yield Member(module, None, name, ())
                yield from _members(node.body, module, name)
    for name, fs in defs.items():
        yield Member(module, cls, name, tuple(fs))


def _stub_params(f: ast.FunctionDef) -> list[P]:
    a = f.args
    pos = [*a.posonlyargs, *a.args]
    pos_defaults = [P.empty] * (len(pos) - len(a.defaults)) + a.defaults
    params = [
        P(arg.arg, P.POSITIONAL_ONLY if i < len(a.posonlyargs) else P.POSITIONAL_OR_KEYWORD, default=d)
        for i, (arg, d) in enumerate(zip(pos, pos_defaults, strict=True))
    ]
    if a.vararg:
        params.append(P(a.vararg.arg, P.VAR_POSITIONAL))
    params += [
        P(arg.arg, P.KEYWORD_ONLY, default=P.empty if d is None else d)
        for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True)
    ]
    if a.kwarg:
        params.append(P(a.kwarg.arg, P.VAR_KEYWORD))
    return params


STUBS = {_module_name(path): ast.parse(path.read_text()) for path in SRC.glob("*-stubs/**/*.pyi")}
MEMBERS = sorted((m for mod, tree in STUBS.items() for m in _members(tree.body, mod)), key=lambda m: m.id)
BASES = {
    node.name: {
        n.id
        for b in node.bases
        if isinstance(n := b.value if isinstance(b, ast.Subscript) else b, ast.Name)  # `Base[T]` → `Base`
    }
    for tree in STUBS.values()
    for node in ast.walk(tree)
    if isinstance(node, ast.ClassDef)
}


def _subclasses(cls: str) -> set[str]:
    """`cls` and all classes in the stubs inheriting from it."""
    return {cls} | {s for c, bases in BASES.items() if cls in bases for s in _subclasses(c)}


# stubs vs runtime


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_private(name: str) -> bool:
    return name.startswith("_") and not _is_dunder(name)


# members that only exist for type checkers
_TYPING_ONLY = {
    # `ArrayLike`, as `numpy` functions fall back to methods like `.sum()`
    "cupyx.scipy.sparse._base._spbase.__array__",
}


def _runtime(member: Member) -> object:
    if member.id in _TYPING_ONLY:
        pytest.skip("typing-only member")
    obj: ModuleType | object = importlib.import_module(member.module)
    for name in filter(None, [member.cls, member.name]):
        if not hasattr(obj, name):
            if _is_private(name):
                pytest.skip(f"private stub-only helper {name!r}")
            pytest.fail(f"{member.id} doesn’t exist at runtime")
        obj = getattr(obj, name)
    return obj


def _default(p: P) -> object:
    """Evaluate a stub default, or return `...` if it can’t be compared."""
    if p.default is P.empty:
        return P.empty
    try:
        return ast.literal_eval(p.default)
    except ValueError:
        return ...


@pytest.mark.parametrize("member", [m for m in MEMBERS if m.defs], ids=lambda m: m.id)
def test_signature_matches_runtime(member: Member) -> None:
    obj = _runtime(member)
    assert callable(obj)
    try:
        sig = inspect.signature(obj)
    except ValueError:
        pytest.skip("no runtime signature (Cython builtin)")
    runtime = list(sig.parameters.values())
    if any(p.kind in {P.VAR_POSITIONAL, P.VAR_KEYWORD} for p in runtime):
        pytest.skip("runtime only takes *args/**kwargs")
    if member.cls and not isinstance(inspect.getattr_static(_runtime_cls(member), member.name), staticmethod):
        runtime = runtime[1:]  # drop `self`

    for stub in member.params():
        if _is_dunder(member.name) and member.name != "__init__":  # called via operators, so only arity matters
            assert len(stub) == len(runtime)
            continue
        if len(member.defs) > 1:  # each overload only needs to be a subset of the runtime signature
            assert {p.name for p in stub} <= {p.name for p in runtime}
            continue
        assert [(p.name, p.kind, p.default is P.empty) for p in stub] == [
            (p.name, p.kind, p.default is P.empty) for p in runtime
        ]
        for s, r in zip(stub, runtime, strict=True):
            if (d := _default(s)) is not ...:
                assert d == r.default, f"default of {s.name!r}"


def _runtime_cls(member: Member) -> type:
    cls: type = getattr(importlib.import_module(member.module), member.cls or "")
    return cls


# tests cover stubs

_OPERATORS: dict[type[ast.AST], str] = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.Div: "__truediv__",
    ast.MatMult: "__matmul__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitAnd: "__and__",
    ast.BitOr: "__or__",
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
}
_BUILTINS = {
    "len": "__len__",
    "iter": "__iter__",
    "next": "__next__",
    "list": "__iter__",
    "int": "__int__",
    "float": "__float__",
}
_CALLABLE_INSTANCES = {
    node.target.id
    for tree in STUBS.values()
    for node in tree.body
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
}


def _reflected(dunder: str) -> str:
    return f"__r{dunder.removeprefix('__')}"


@dataclass
class Usage:
    names: set[str]
    calls: defaultdict[str, list[ast.Call]]
    dunders: set[str]

    @classmethod
    def collect(cls, paths: Iterable[Path]) -> Usage:
        usage = cls(set(), defaultdict(list), set())
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                usage.add(node)
        return usage

    def add(self, node: ast.AST) -> None:
        match node:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                self.names.add(name)
            case ast.BinOp(left=ast.Constant(), op=op) if type(op) in _OPERATORS:
                self.dunders.add(_reflected(_OPERATORS[type(op)]))
            case ast.BinOp(op=op) | ast.Compare(ops=[op, *_]) if type(op) in _OPERATORS:
                self.dunders.add(_OPERATORS[type(op)])
            case ast.UnaryOp(op=ast.Invert()):
                self.dunders.add("__invert__")
            case ast.Subscript(ctx=ctx):
                self.dunders.add("__setitem__" if isinstance(ctx, ast.Store) else "__getitem__")
            case ast.For():
                self.dunders.add("__iter__")
            case _:
                pass
        match node:
            case ast.Call(func=ast.Name(id=name) | ast.Attribute(attr=name)):
                self.calls[name].append(node)
                if name in _BUILTINS:
                    self.dunders.add(_BUILTINS[name])
                # shortcut: any `np.…(…)` call counts, since we don’t know argument types here
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "np" and node.args:
                        self.dunders.add("__array__")
            case _:
                pass


@pytest.fixture(scope="module")
def usage() -> Usage:
    return Usage.collect(p for p in HERE.glob("test_*.py") if p != Path(__file__))


def _uncovered_params(member: Member, calls: list[ast.Call]) -> set[str]:
    # shortcut: matches calls by callee name only (`a.sum` also covers `csr.sum`), use mypy’s reference info if too lax
    uncovered = set()
    for params in member.params():
        for i, p in enumerate(params):
            if p.kind in {P.VAR_POSITIONAL, P.VAR_KEYWORD}:
                continue
            by_kw = any(kw.arg == p.name for c in calls for kw in c.keywords)
            by_pos = p.kind is not P.KEYWORD_ONLY and any(len(c.args) > i for c in calls)
            if not (by_kw or by_pos):
                uncovered.add(p.name)
    return uncovered


@pytest.mark.parametrize("member", MEMBERS, ids=lambda m: m.id)
def test_covered(member: Member, usage: Usage) -> None:
    name = member.name
    if member.cls is None and _is_private(name):
        pytest.skip("private helper classes are covered through their public subclasses")
    if name == "__init__":
        calls = [c for cls in _subclasses(member.cls or "") for c in usage.calls[cls]]
        assert calls, f"{member.cls} is never instantiated"
    elif name == "__call__":
        calls = [c for n in _CALLABLE_INSTANCES for c in usage.calls[n]]
        assert calls, f"{member.cls} instances are never called"
    elif _is_dunder(name):
        operators = {*_OPERATORS.values(), *map(_reflected, _OPERATORS.values())}
        known = {*operators, *_BUILTINS.values(), "__getitem__", "__setitem__", "__array__", "__invert__"}
        assert name in known, f"don’t know how {name} is used (typo, or extend `_OPERATORS`/`_BUILTINS`?)"
        assert name in usage.dunders, f"{name} is never used"
        return
    else:
        assert name in usage.names, f"{name} is never used"
        calls = usage.calls[name]
    assert not (uncovered := _uncovered_params(member, calls)), f"parameters never passed: {sorted(uncovered)}"
