import os
os.environ["KINETO_LOG_LEVEL"] = "5"

import ast
import dis
import sys
import time
import inspect
from pathlib import Path
from contextlib import ExitStack
from collections import defaultdict

import torch

from captural.path import PathChecker


class LineNumberVisitor(ast.NodeVisitor):
    def __init__(self, line_number):
        self.line_number = line_number
        self.node = []

    def visit(self, node):
        if hasattr(node, 'lineno') and node.lineno == self.line_number:
            self.node.append(node)
        return super().visit(node)


class Captural:
    def __init__(self, paths=["."]):
        self.path_checker = PathChecker(paths)
        self._capture_data = []
        # self._profiler = None
        # self._context_stack = None

        self._is_assigned = False
        self._assign_frame = None
        self._assign_cnt = -1
        self._assign_node = None
        self._assign_source_file = None
        self._assign_source_line = None
        self._assign_source_func = None

        self._file_ast_cache = {}

        self._skipped_files = defaultdict(int)

    def _chk_general_function(self, frame):
        source_file = frame.f_code.co_filename
        source_line = frame.f_lineno
        source_func = frame.f_code.co_name

        if source_func[0] == "<":
            return False
        elif source_func[-1] == ">":
            return False
        elif not Path(source_file).exists():
            return False
        elif source_file[0] == "<":
            return False
        elif source_file[-1] == ">":
            return False
        return True

    def _get_ast_node(self, file_path:str, line_number:int):
        if file_path not in self._file_ast_cache:
            with open(file_path, "r") as f:
                self._file_ast_cache[file_path] = ast.parse(f.read())
        file_ast_tree = self._file_ast_cache[file_path]

        visitor = LineNumberVisitor(line_number)
        visitor.visit(file_ast_tree)

        return visitor.node

    def _get_current_assign_node(self, source_file:str, source_line:int):
        if not Path(source_file).exists():
            return None
        for node in self._get_ast_node(source_file, source_line):
            if isinstance(node, ast.Assign):
                return node
        return None

    def _get_trace_name(self, frame):
        co = frame.f_code
        lineno = frame.f_lineno
        func_name = co.co_name
        filename = co.co_filename
        return f"{filename}:{lineno}:{func_name}"

    def trace_calls(self, frame, event, arg):
        # print(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
        if event == 'call':
            if self.path_checker.check(frame.f_code.co_filename):
                frame.f_trace_lines = False
                frame.f_trace_opcodes = True
                return self.trace_lines
            else:
                self._skipped_files[frame.f_code.co_filename] += 1
        return None

    def trace_lines(self, frame, event, arg):
        if event != "opcode":
            return

        torch.cuda.synchronize()

        # self._context_stack.__exit__(None, None, None)

        def _init_is_compute(event_name):
            if "aten::" in event_name or "cudaLaunchKernel" in event_name:
                return True
            return False

        is_compute = False

        # for event in self._profiler.events():
        #     is_compute = is_compute or _init_is_compute(event.name)

        source_file = frame.f_code.co_filename
        source_line = frame.f_lineno
        source_func = frame.f_code.co_name
        source_opcode = frame.f_code.co_code[frame.f_lasti]
        source_opname = dis.opname[source_opcode]

        if self._is_assigned:
            def _tensor_to_shape(data):
                if torch.is_tensor(data):
                    return "Tensor[" + ", ".join(map(str, data.shape)) + "]"
                elif isinstance(data, tuple):
                    return "(" + ", ".join(_tensor_to_shape(_data) for _data in data) + ")"
                elif isinstance(data, list):
                    return "[" + ", ".join(_tensor_to_shape(_data) for _data in data) + "]"
                elif isinstance(data, dict):
                    return "{" + ", ".join(f"{k}: {_tensor_to_shape(v)}" for k, v in data.items()) + "}"
                elif data is None:
                    return "None"
                return str(data)

            def _get_raw_data(node):
                if isinstance(node, ast.Name):
                    val = frame.f_locals[node.id]
                    return f"{node.id}: {_tensor_to_shape(val)}"
                elif isinstance(node, ast.Subscript):
                    key = ast.unparse(node.slice)
                    val = frame.f_locals[node.value.id][frame.f_locals[key]]
                    return f"{node.value.id}[{key}]: {_tensor_to_shape(val)}"
                elif isinstance(node, ast.Attribute):
                    val = getattr(frame.f_locals[node.value.id], node.attr)
                    return f"{node.value.id}.{node.attr}: {_tensor_to_shape(val)}"
                elif isinstance(node, ast.Tuple):
                    return "(" + ", ".join(_get_raw_data(_node) for _node in node.elts) + ")"
                elif isinstance(node, ast.List):
                    return "[" + ", ".join(_get_raw_data(_node) for _node in node.elts) + "]"
                else:
                    return "None"

            target_value = _get_raw_data(self._assign_node.targets[0])

            self._capture_data.append({
                "is_assign": True,
                "target_value": target_value,
                "source_file": self._assign_source_file,
                "source_line": self._assign_source_line,
                "source_func": self._assign_source_func,
            })

            self._is_assigned = False
            self._assign_cnt = -1
            self._assign_node = None
            self._assign_source_file = None
            self._assign_source_line = None
            self._assign_source_func = None
        else:
            if source_opname.startswith("STORE_"):
                if self._assign_node is None or self._assign_frame != frame:
                    assign_node = self._get_current_assign_node(source_file, source_line)
                    if assign_node is not None:
                        def _calc_assign_cnt(node):
                            if isinstance(node, ast.Assign):
                                return sum(map(_calc_assign_cnt, node.targets))
                            elif isinstance(node, ast.Tuple):
                                return sum(map(_calc_assign_cnt, node.elts))
                            elif isinstance(node, ast.List):
                                return sum(map(_calc_assign_cnt, node.elts))
                            elif isinstance(node, ast.Name):
                                return 1
                            elif isinstance(node, ast.Attribute):
                                return 1
                            elif isinstance(node, ast.Subscript):
                                return 1
                            return 0
                        # print(ast.dump(assign_node), _calc_assign_cnt(assign_node))
                        self._assign_frame = frame
                        self._assign_cnt = _calc_assign_cnt(assign_node)
                        self._assign_node = assign_node
                        self._assign_source_file = source_file
                        self._assign_source_line = source_line
                        self._assign_source_func = source_func

                        # print(f"Assign: {source_file}:{source_line} ({source_func}), {ast.dump(self._assign_node)} {_calc_assign_cnt(assign_node)}")
                    else:  # maybe a `with` statement
                        pass

                if self._chk_general_function(frame):
                    self._assign_cnt -= 1

                if self._assign_cnt == 0:
                    self._is_assigned = True

            if self._chk_general_function(frame):
                self._capture_data.append({
                    "is_assign": False,
                    "source_file": source_file,
                    "source_line": source_line,
                    "source_func": source_func,
                })

        # with ExitStack() as stack:
        #     self._profiler = torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             # torch.profiler.ProfilerActivity.CUDA,
        #         ],
        #         with_stack=True,
        #     )
        #     stack.enter_context(self._profiler)
        #     self._context_stack = stack.pop_all()

        torch.cuda.synchronize()
        self._start_time = time.time()

    def print_stats(self):
        print("Captural Profiler Stats:")
        source_lines = []
        current_source_file = None
        current_source_line_start = None
        current_source_line_end = None
        current_source_func = None
        current_source_line_desc = dict()
        for trace_event in self._capture_data:
            is_assign = trace_event["is_assign"]
            # print(trace_event)
            if (
                current_source_func != trace_event["source_func"]
                or current_source_file != trace_event["source_file"]
            ):  # todo: check using frame
                if current_source_file is not None:
                    print(f"Source: {current_source_file}:{current_source_line_start}-{current_source_line_end} ({current_source_func})")
                    with open(current_source_file, "r") as f:
                        source_lines = f.readlines()
                    for i, line in enumerate(source_lines):
                        lineno = i + 1
                        if lineno >= current_source_line_start and lineno <= current_source_line_end:
                            print(f"{lineno}: {line.rstrip().ljust(100)[:100]} | {current_source_line_desc.get(lineno, '')}")
                    print()

                current_source_file = trace_event["source_file"]
                current_source_line_start = trace_event["source_line"]
                current_source_line_end = trace_event["source_line"]
                current_source_func = trace_event["source_func"]
                current_source_line_desc = dict()
            else:
                current_source_line_end = max(current_source_line_end, trace_event["source_line"])
                if is_assign:
                    current_source_line_desc[trace_event["source_line"]] = trace_event["target_value"]

        for source_file, cnt in self._skipped_files.items():
            print(f"Skipped: {source_file} ({cnt})")

    def __enter__(self):
        # with ExitStack() as stack:
        #     self._profiler = torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             # torch.profiler.ProfilerActivity.CUDA,
        #         ],
        #         with_stack=True,
        #     )
        #     stack.enter_context(self._profiler)
        #     self._context_stack = stack.pop_all()
        sys.settrace(self.trace_calls)
        caller_frame = sys._getframe(1)
        self.caller_frame_f_trace_bkp = caller_frame.f_trace
        caller_frame.f_trace = self.trace_calls(caller_frame, 'call', None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.settrace(None)
        caller_frame = sys._getframe(1)
        caller_frame.f_trace = self.caller_frame_f_trace_bkp

        # self._context_stack.__exit__(exc_type, exc_val, exc_tb)
