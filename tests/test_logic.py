import asyncio
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from comfy.cli_args import args

args.cpu = True

import execution
import nodes
import torch
from comfy_api.latest import io
from custom_nodes.Min_node import comfy_entrypoint
from custom_nodes.Min_node.logic import (
    LOGIC_NODES,
    LogicAnd,
    LogicNand,
    LogicNor,
    LogicNot,
    LogicOr,
    LogicXnor,
    LogicXor,
    Mux2x1,
    Mux4x1,
)


class LogicTests(unittest.TestCase):
    def test_truth_tables_return_integers(self):
        tables = {
            LogicAnd: (0, 0, 0, 1),
            LogicOr: (0, 1, 1, 1),
            LogicXor: (0, 1, 1, 0),
            LogicNand: (1, 1, 1, 0),
            LogicNor: (1, 0, 0, 0),
            LogicXnor: (1, 0, 0, 1),
        }
        for gate, expected in tables.items():
            for (a, b), value in zip(itertools.product((0, 1), repeat=2), expected):
                with self.subTest(gate=gate.__name__, a=a, b=b):
                    result = gate.execute(a, b)[0]
                    self.assertIs(type(result), int)
                    self.assertEqual(result, value)
        for a, expected in ((0, 1), (1, 0)):
            result = LogicNot.execute(a)[0]
            self.assertIs(type(result), int)
            self.assertEqual(result, expected)

    def test_nonzero_integers_are_true(self):
        for a, b in itertools.product((0, -7, 9), repeat=2):
            for gate in (LogicAnd, LogicOr, LogicXor, LogicNand, LogicNor, LogicXnor):
                with self.subTest(gate=gate.__name__, a=a, b=b):
                    self.assertEqual(gate.execute(a, b)[0], gate.execute(int(bool(a)), int(bool(b)))[0])
        self.assertEqual(LogicNot.execute(-7)[0], 0)

    def test_extension_registers_new_and_existing_nodes(self):
        extension = asyncio.run(comfy_entrypoint())
        registered = asyncio.run(extension.get_node_list())
        ids = [node.define_schema().node_id for node in registered]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(registered), 27)
        self.assertTrue(set(LOGIC_NODES).issubset(registered))
        self.assertIn("MinNamedGet", ids)
        for node in LOGIC_NODES:
            node.GET_SCHEMA()

    def test_mux_preserves_each_selected_object(self):
        payloads = [torch.zeros(1, 2, 2, 3), "text", object(), {"samples": torch.zeros(1)}, 0, False, None]
        for mux in (Mux2x1, Mux4x1):
            for select, payload in itertools.product(range(mux.input_count), payloads):
                with self.subTest(mux=mux.__name__, select=select, type=type(payload).__name__):
                    inputs = {f"input{i}": object() for i in range(mux.input_count)}
                    inputs[f"input{select}"] = payload
                    self.assertIs(mux.execute(select, **inputs)[0], payload)

    def test_mux_requests_only_selected_lazy_input(self):
        for mux in (Mux2x1, Mux4x1):
            for select in range(mux.input_count):
                inputs = {f"input{i}": None for i in range(mux.input_count)}
                self.assertEqual(mux.check_lazy_status(select, **inputs), [f"input{select}"])
                inputs[f"input{select}"] = "ready"
                self.assertEqual(mux.check_lazy_status(select, **inputs), [])
                self.assertEqual(mux.execute(select, **{f"input{select}": "ready"})[0], "ready")

    def test_mux_reports_missing_selection_and_invalid_index(self):
        for mux in (Mux2x1, Mux4x1):
            for method in (mux.check_lazy_status, mux.execute):
                with self.assertRaisesRegex(ValueError, "not connected"):
                    method(select=0, input1="unused")
                for select in (-1, mux.input_count):
                    with self.assertRaisesRegex(ValueError, "select must be"):
                        method(select=select, input0="value")


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def run_mux_graph(self, mux, select, disconnected=False, chained_selector=False):
        data_types = (io.Image, io.String, io.Clip, io.Custom("MIN_TEST_CUSTOM"))
        payloads = (torch.zeros(1, 2, 2, 3), "hello", object(), {"custom": object()})
        visited = []
        captured = []

        def make_source(index):
            class Source(io.ComfyNode):
                @classmethod
                def define_schema(cls):
                    return io.Schema(node_id=f"MinTestSource{index}", inputs=[], outputs=[data_types[index].Output()])

                @classmethod
                def execute(cls):
                    if index != select:
                        raise AssertionError("An unselected MUX branch was executed")
                    visited.append(index)
                    return io.NodeOutput(payloads[index])

            return Source

        class Sink(io.ComfyNode):
            @classmethod
            def define_schema(cls):
                return io.Schema(
                    node_id="MinTestSink",
                    inputs=[data_types[select].Input("value")],
                    outputs=[],
                    is_output_node=True,
                )

            @classmethod
            def execute(cls, value):
                captured.append(value)
                return io.NodeOutput()

        classes = {node.define_schema().node_id: node for node in LOGIC_NODES}
        classes["MinTestSink"] = Sink
        prompt = {}
        mux_inputs = {"select": select}
        for index in range(mux.input_count):
            if disconnected and index != select:
                continue
            source = make_source(index)
            source_id = source.define_schema().node_id
            classes[source_id] = source
            prompt[f"source{index}"] = {"class_type": source_id, "inputs": {}}
            mux_inputs[f"input{index}"] = [f"source{index}", 0]
        if chained_selector:
            prompt["selector"] = {"class_type": "MinLogicNOT", "inputs": {"a": 0}}
            mux_inputs["select"] = ["selector", 0]
        prompt["mux"] = {"class_type": mux.define_schema().node_id, "inputs": mux_inputs}
        prompt["sink"] = {"class_type": "MinTestSink", "inputs": {"value": ["mux", 0]}}

        with patch.dict(nodes.NODE_CLASS_MAPPINGS, classes):
            valid, _, outputs, errors = await execution.validate_prompt("min-mux-test", prompt, None)
            self.assertTrue(valid, errors)
            server = SimpleNamespace(client_id=None, send_sync=Mock())
            executor = execution.PromptExecutor(
                server,
                cache_type=execution.CacheType.NONE,
                cache_args={"ram": 0, "ram_inactive": 0},
            )
            await executor.execute_async(prompt, "min-mux-test", execute_outputs=outputs)
            self.assertTrue(executor.success, executor.status_messages)
        self.assertEqual(visited, [select])
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0], payloads[select])

    async def test_typed_links_and_lazy_execution(self):
        for mux in (Mux2x1, Mux4x1):
            for select in range(mux.input_count):
                with self.subTest(mux=mux.__name__, select=select):
                    await self.run_mux_graph(mux, select)

    async def test_unselected_inputs_can_be_disconnected(self):
        await self.run_mux_graph(Mux2x1, 1, disconnected=True)
        await self.run_mux_graph(Mux4x1, 3, disconnected=True)

    async def test_gate_output_can_control_mux(self):
        await self.run_mux_graph(Mux2x1, 1, chained_selector=True)


if __name__ == "__main__":
    unittest.main()
