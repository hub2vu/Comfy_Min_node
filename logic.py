from comfy_api.latest import io


class _BinaryGate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=f"MinLogic{cls.operation}",
            display_name=cls.operation,
            category="Min_node/Logic",
            description=f"Logical {cls.operation}. Zero is false; nonzero integers are true. Returns integer 0 or 1.",
            inputs=[
                io.Int.Input("a", default=0, min=0, max=1),
                io.Int.Input("b", default=0, min=0, max=1),
            ],
            outputs=[io.Int.Output("result")],
        )


class LogicAnd(_BinaryGate):
    operation = "AND"

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(int(bool(a) and bool(b)))


class LogicOr(_BinaryGate):
    operation = "OR"

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(int(bool(a) or bool(b)))


class LogicNot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinLogicNOT",
            display_name="NOT",
            category="Min_node/Logic",
            description="Logical NOT. Zero is false; nonzero integers are true. Returns integer 0 or 1.",
            inputs=[io.Int.Input("a", default=0, min=0, max=1)],
            outputs=[io.Int.Output("result")],
        )

    @classmethod
    def execute(cls, a: int) -> io.NodeOutput:
        return io.NodeOutput(int(not a))


class LogicXor(_BinaryGate):
    operation = "XOR"

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(int(bool(a) != bool(b)))


class LogicNand(_BinaryGate):
    operation = "NAND"

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(int(not (a and b)))


class LogicNor(_BinaryGate):
    operation = "NOR"

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(int(not (a or b)))


class LogicXnor(_BinaryGate):
    operation = "XNOR"

    @classmethod
    def execute(cls, a: int, b: int) -> io.NodeOutput:
        return io.NodeOutput(int(bool(a) == bool(b)))


class _Mux(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=f"MinMux{cls.input_count}x1",
            display_name=f"MUX {cls.input_count}x1",
            category="Min_node/Logic",
            description="Routes input0, input1, etc. by the zero-based select index. Accepts any data type and evaluates only the selected input.",
            inputs=[
                io.Int.Input("select", default=0, min=0, max=cls.input_count - 1),
                *[
                    io.AnyType.Input(f"input{i}", optional=True, lazy=True)
                    for i in range(cls.input_count)
                ],
            ],
            outputs=[io.AnyType.Output("output")],
        )

    @classmethod
    def _selected_input(cls, select: int, inputs: dict) -> str:
        if select not in range(cls.input_count):
            raise ValueError(f"select must be between 0 and {cls.input_count - 1}.")
        name = f"input{select}"
        if name not in inputs:
            raise ValueError(f"Selected input '{name}' is not connected.")
        return name

    @classmethod
    def check_lazy_status(cls, select: int, **inputs) -> list[str]:
        name = cls._selected_input(select, inputs)
        return [name] if inputs[name] is None else []

    @classmethod
    def execute(cls, select: int, **inputs) -> io.NodeOutput:
        return io.NodeOutput(inputs[cls._selected_input(select, inputs)])


class Mux2x1(_Mux):
    input_count = 2


class Mux4x1(_Mux):
    input_count = 4


LOGIC_NODES = [LogicAnd, LogicOr, LogicNot, LogicXor, LogicNand, LogicNor, LogicXnor, Mux2x1, Mux4x1]
