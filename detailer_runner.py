import ast
import copy
import inspect
import textwrap
import tokenize
from types import CodeType

from .detailer_inspection import DetailerInspection


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    return ""


def _calls(node, name):
    return any(isinstance(item, ast.Call) and _call_name(item.func) == name for item in ast.walk(node))


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)


def _boundary(phase, value="None"):
    node = ast.parse(f"yield ({phase!r}, {value})").body[0]
    node._min_boundary = True
    return node


def _selectable_boundary(phase, value, skip):
    node = ast.parse(f"if (yield ({phase!r}, {value})) is False:\n{textwrap.indent(skip, '    ')}").body[0]
    node._min_boundary = True
    return node


def _insert(body, match, before=(), after=()):
    indices = [i for i, statement in enumerate(body) if match(statement)]
    if len(indices) != 1:
        raise RuntimeError("The installed Impact Pack has an unsupported FaceDetailer layout; no altered implementation will be used.")
    index = indices[0]
    body[index:index + 1] = [*before, body[index], *after]


class _DelegateCalls(ast.NodeTransformer):
    targets = {
        "FaceDetailer.enhance_face": "_min_enhance_face",
        "DetailerForEach.do_detail": "_min_do_detail",
        "core.enhance_detail": "_min_enhance_detail",
    }

    def __init__(self):
        self.count = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        name = _call_name(node.func)
        if name not in self.targets:
            return node
        original = copy.deepcopy(node)
        node.func = ast.Name(id=self.targets[name], ctx=ast.Load())
        delegated = ast.YieldFrom(value=node)
        delegated._min_original = original
        self.count += 1
        return ast.copy_location(delegated, node)


class _RestoreOriginal(ast.NodeTransformer):
    def visit(self, node):
        return None if getattr(node, "_min_boundary", False) else super().visit(node)

    def visit_YieldFrom(self, node):
        return copy.deepcopy(node._min_original) if hasattr(node, "_min_original") else self.generic_visit(node)


def _code_signature(code):
    constants = tuple(_code_signature(value) if isinstance(value, CodeType) else value for value in code.co_consts)
    return code.co_code, constants, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_flags


def _source_matches(function, filename):
    with tokenize.open(filename) as source:
        pending = [compile(source.read(), filename, "exec", dont_inherit=True)]
    expected = _code_signature(function.__code__)
    while pending:
        code = pending.pop()
        if code.co_qualname == function.__qualname__ and _code_signature(code) == expected:
            return True
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
    return False


def _split_function(function, kind):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = tree.body[0]
    if function.__closure__ or any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(definition)):
        raise RuntimeError("This FaceDetailer implementation uses closure/global writes and cannot be split unchanged.")
    if any(not isinstance(decorator, ast.Name) or decorator.id not in ("staticmethod", "classmethod") for decorator in definition.decorator_list):
        raise RuntimeError("This FaceDetailer implementation has an unsupported function decorator.")
    definition.decorator_list = []
    original = copy.deepcopy(tree)
    filename = inspect.getsourcefile(function)
    if not _source_matches(function, filename):
        raise RuntimeError("Impact Pack source and loaded code differ. Restart ComfyUI before using the split FaceDetailer.")
    if kind == "face":
        definition.body.insert(0, _boundary("before_detect"))
        _insert(
            definition.body,
            lambda node: isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and _call_name(node.value.func) == "bbox_detector.setAux"
            and ast.literal_eval(node.value.args[0]) is None,
            after=[_boundary("detected", "segs")],
        )
        _insert(definition.body, lambda node: isinstance(node, ast.If) and _calls(node, "core.make_sam_mask"),
                after=[_boundary("masked", "segs")])
        _insert(definition.body, lambda node: _assigns(node, "mask"), before=[_boundary("finish")])
    elif kind == "detail":
        loops = [node for node in definition.body if isinstance(node, ast.For) and _calls(node.iter, "enumerate")]
        if len(loops) != 1:
            raise RuntimeError("The installed Impact Pack FaceDetailer loop is unsupported.")
        loop = loops[0]
        loop.body.insert(0, _boundary("before_crop"))
        branch = [node for node in loop.body if isinstance(node, ast.If) and _calls(node, "core.enhance_detail")]
        if len(branch) != 1 or not branch[0].orelse:
            raise RuntimeError("The installed Impact Pack FaceDetailer sampling branch is unsupported.")
        branch[0].orelse.insert(0, _boundary("cropped", "cropped_image"))
        _insert(loop.body, lambda node: node is branch[0], after=[
            _selectable_boundary("regenerated", "enhanced_image", "enhanced_image = None\ncnet_pils = None"),
        ])
        _insert(loop.body, lambda node: _assigns(node, "mask") and _calls(node, "utils.tensor_gaussian_blur_mask"),
                after=[_boundary("crop_input", "cropped_image")])
        _insert(loop.body, lambda node: _assigns(node, "seg_seed"), after=[_boundary("region_ready")])
        loop.body.append(_boundary("composited"))
        _insert(definition.body, lambda node: _assigns(node, "image_tensor"), before=[_boundary("finish")])
    elif kind == "core":
        _insert(definition.body, lambda node: _assigns(node, "cnet_pils"),
                before=[_selectable_boundary("cropped", "upscaled_image", "return None, None")])

    delegate = _DelegateCalls()
    tree = delegate.visit(tree)
    expected = 0 if kind == "core" else 1
    if delegate.count != expected:
        raise RuntimeError("The installed Impact Pack FaceDetailer call chain is unsupported.")
    restored = _RestoreOriginal().visit(copy.deepcopy(tree))
    if ast.dump(restored) != ast.dump(original):
        raise RuntimeError("FaceDetailer split would change the original operation statements.")
    namespace = function.__globals__.copy()
    ast.fix_missing_locations(tree)
    exec(compile(tree, filename, "exec"), namespace)
    return namespace[definition.name], namespace


class DetailerRun:
    """A single FaceDetailer execution, paused only at the five node boundaries."""

    def __init__(self, face_detailer, parameters):
        module = inspect.getmodule(face_detailer)
        functions = (
            (face_detailer.doit, "outer", "_min_doit"),
            (face_detailer.enhance_face, "face", "_min_enhance_face"),
            (module.DetailerForEach.do_detail, "detail", "_min_do_detail"),
            (module.core.enhance_detail, "core", "_min_enhance_detail"),
        )
        compiled = {}
        namespaces = []
        for function, kind, name in functions:
            compiled[name], namespace = _split_function(function, kind)
            namespaces.append(namespace)
        for namespace in namespaces:
            namespace.update(compiled)
        self.parameters = parameters
        self.generator = compiled["_min_doit"](face_detailer(), **parameters)
        self.phase = None
        self.value = None
        self.result = None
        self.inspection = DetailerInspection(self)
        self.regeneration_options = {"selected_indices": "all", "preview_only": False}

    def frames(self):
        frames = {}
        generator = self.generator
        while inspect.isgenerator(generator):
            if generator.gi_frame is not None:
                frames[generator.gi_code.co_name] = generator.gi_frame.f_locals
            generator = generator.gi_yieldfrom
        return frames

    def advance(self, decision=None):
        try:
            self.phase, self.value = next(self.generator) if decision is None else self.generator.send(decision)
        except StopIteration as finished:
            self.result = finished.value
            self.generator = None
            self.phase, self.value = "complete", None
        self.inspection.observe(self.phase)
        return self.phase

    def require(self, *phases):
        if self.phase not in phases:
            raise ValueError(f"FaceDetailer stage order is invalid: {self.phase!r}. Connect stages 1 -> 2 -> 3 -> 4 -> 5 once, without branching the state sockets.")
