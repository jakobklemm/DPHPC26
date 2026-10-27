"""Small host-only checks for plan semantics and a direct CMake benchmark."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from filter_llama_performance_flags import find_cuda_compiler

spec = importlib.util.spec_from_file_location("runner", ROOT / "llama_gpu_bench-random.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.options = {
            "cmake": {"GGML_NATIVE": ["ON", "OFF"], "GGML_AVX": ["ON", "OFF"]},
            "device": {"GGML_CUDA": ["ON"], "GGML_CUDA_FORCE_MMQ": ["ON", "OFF"],
                       "GGML_CUDA_FORCE_CUBLAS": ["ON", "OFF"]},
            "compiler": {"-O": ["-O3", "-O2"],
                         "-fvectorize": ["-fvectorize", "-fno-vectorize"]},
            "invalid_combinations": [
                {"GGML_CUDA_FORCE_MMQ": "ON", "GGML_CUDA_FORCE_CUBLAS": "ON"}
            ],
        }

    def test_complete_sweep_and_cpu_only(self):
        options = {**self.options, "device": {}, "invalid_combinations": []}
        plan = runner.execution_plan(options, "cpu")
        self.assertEqual(len(plan), 4)
        self.assertEqual(len({json.dumps(item["config"], sort_keys=True) for item in plan}), 4)
        for item in plan:
            config = item["config"]
            self.assertEqual(config["cmake"]["GGML_CUDA"], "OFF")
            self.assertEqual(config["cmake"]["GGML_VULKAN"], "OFF")
            self.assertEqual(len(config["cmake"]), 4)
            self.assertEqual(len(config["compiler_flags"]), 2)
            self.assertEqual(sum(value == "ON" for value in config["cmake"].values()) +
                             sum(config["compiler_flags"][key] != runner.off_value(key)
                                 for key in config["compiler_flags"]), 1)

    def test_invalid_subsets_are_skipped_before_execution(self):
        plan = runner.execution_plan(self.options, "cuda", "86", "subset", 8, 2)
        self.assertEqual(len(plan), 8)
        self.assertEqual(plan, runner.execution_plan(self.options, "cuda", "86", "subset", 8, 2))
        for item in plan:
            config = item["config"]
            self.assertEqual(config["cmake"]["CMAKE_CUDA_ARCHITECTURES"], "86")
            self.assertFalse(config["cmake"]["GGML_CUDA_FORCE_MMQ"] == "ON" and
                             config["cmake"]["GGML_CUDA_FORCE_CUBLAS"] == "ON")
            self.assertEqual(len(config["compiler_flags"]), 2)

    def test_default_subset_has_exactly_seven_enabled_flags(self):
        options = {**self.options, "cmake": {
            **self.options["cmake"], **{f"GGML_TEST_{n}": ["ON", "OFF"] for n in range(8)}},
            "device": {}, "invalid_combinations": []}
        plan = runner.execution_plan(options, "cpu", mode="subset", count=3)
        self.assertEqual(len(plan), 3)
        for item in plan:
            config = item["config"]
            enabled = sum(value == "ON" for value in config["cmake"].values())
            enabled += sum(value != runner.off_value(name)
                           for name, value in config["compiler_flags"].items())
            self.assertEqual(enabled, 7)

    def test_vulkan_runtime_flags_have_explicit_off_values(self):
        options = {**self.options, "device": {"GGML_VULKAN": ["ON"],
                                              "GGML_VK_DISABLE_F16": ["1", "0"]}}
        plan = runner.execution_plan(options, "vulkan")
        self.assertEqual({item["config"]["runtime"]["GGML_VK_DISABLE_F16"] for item in plan},
                         {"0", "1"})
        self.assertTrue(all(item["config"]["cmake"]["GGML_CUDA"] == "OFF" for item in plan))

    def test_zero_subsets(self):
        self.assertEqual(runner.execution_plan(self.options, "cuda", "86", "subset", 0), [])


class PathTests(unittest.TestCase):
    def test_cuda_compiler_can_be_set_with_cudacxx(self):
        with tempfile.TemporaryDirectory() as directory:
            compiler = Path(directory) / "nvcc"
            compiler.write_text("#!/bin/sh\nexit 0\n")
            compiler.chmod(0o755)
            with patch.dict(os.environ, {"CUDACXX": str(compiler)}):
                self.assertEqual(find_cuda_compiler(), str(compiler))

    def test_script_local_defaults_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for filename in ("llama_gpu_bench-random.py", "filter_llama_performance_flags.py"):
                shutil.copy2(ROOT / filename, base / filename)
            source = base / "llama.cpp"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("project(tiny)\n")
            (base / "model.gguf").write_bytes(b"model")
            catalog = {"cpu": {"GGML_NATIVE": ["ON", "OFF"]}, "compiler": {},
                       "device": {}, "invalid_combinations": []}
            (base / "flags.json").write_text(json.dumps(catalog))
            command = [sys.executable, str(base / "llama_gpu_bench-random.py"), "sweep",
                       "--device", "cpu", "--dry-run"]
            result = subprocess.run(command, cwd=ROOT.parent, check=True,
                                    capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout.split("\n", 1)[1])[0]["config"]
                             ["cmake"]["GGML_NATIVE"], "ON")

            override = base / "alternative.json"
            catalog["cpu"]["GGML_CPU_REPACK"] = ["ON", "OFF"]
            override.write_text(json.dumps(catalog))
            result = subprocess.run([*command, "--flags-file", str(override)], cwd=ROOT.parent,
                                    check=True, capture_output=True, text=True)
            self.assertEqual(len(json.loads(result.stdout.split("\n", 1)[1])), 2)

    def test_profile_checks_rules_from_provided_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("project(tiny)\n")
            model = base / "model.gguf"
            model.write_bytes(b"model")
            catalog = base / "custom-flags.json"
            catalog.write_text(json.dumps({"invalid_combinations": [{"GGML_CUDA": "OFF"}]}))
            result = subprocess.run([sys.executable, str(ROOT / "llama_gpu_bench-profile.py"),
                                     "--source", str(source), "--model", str(model),
                                     "--flags-file", str(catalog), "--device", "cpu",
                                     "--gpu-layers", "0", "--cmake-arg", "GGML_CUDA=OFF"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid CMake flag combination", result.stderr)

    def test_cuda_preflight_reports_missing_nvcc_before_any_build(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text("project(tiny)\n")
            model = base / "model.gguf"
            model.write_bytes(b"model")
            result = subprocess.run([
                sys.executable, str(ROOT / "llama_gpu_bench-random.py"), "subset",
                "--source", str(source), "--model", str(model), "--device", "cuda",
                "--cuda-arch", "75", "--count", "1", "--flags-per-build", "1",
                "--results-dir", str(base / "results"),
            ], capture_output=True, text=True,
               env={**os.environ, "CUDACXX": "/no/such/nvcc"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA builds require nvcc", result.stderr)
            self.assertFalse((base / "results").exists())


class LocalBuildTests(unittest.TestCase):
    def test_direct_cpu_build_and_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.18)\nproject(tiny LANGUAGES C CXX)\n'
                'add_executable(llama-bench main.cpp)\n'
                'set_target_properties(llama-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n'
            )
            (source / "main.cpp").write_text(
                '#include <cstdlib>\n#include <iostream>\nint main() { '
                'if (std::getenv("LLAMA_BENCH_TEST_FAIL")) return 7; '
                'std::cout << "[{\\"backends\\":\\"CPU\\",'
                '\\"avg_ns\\":1000,\\"n_prompt\\":1,\\"n_gen\\":0}]"; }\n'
            )
            model = base / "tiny.gguf"
            model.write_bytes(b"test model placeholder")
            command = [sys.executable, str(ROOT / "llama_gpu_bench-profile.py"),
                       "--source", str(source), "--model", str(model), "--device", "cpu",
                       "--build-dir", str(base / "build"),
                       "--results-dir", str(base / "results"), "--workers", "2",
                       "--cmake-arg", "GGML_CUDA=OFF", "--cmake-arg", "GGML_VULKAN=OFF",
                       "--compiler-flag=-O0"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            run = next((base / "results").iterdir())
            build_record = json.loads((run / "build.json").read_text())
            self.assertEqual(build_record["status"], "completed")
            self.assertEqual(build_record["flags_file"], str(ROOT / "flags.json"))
            self.assertEqual(build_record["benchmark_command"][4], "0")
            self.assertTrue(all("-DNDEBUG" in flag for flag in build_record["configure_command"]
                                if flag.startswith(("-DCMAKE_C_FLAGS_RELEASE=",
                                                    "-DCMAKE_CXX_FLAGS_RELEASE="))))
            self.assertEqual(json.loads((run / "benchmark.json").read_text())[0]["backends"], "CPU")
            self.assertNotIn("docker", (run / "build.log").read_text().lower())
            sweep = [sys.executable, str(ROOT / "llama_gpu_bench-random.py"), "subset",
                     "--source", str(source), "--model", str(model), "--device", "cpu",
                     "--count", "1", "--flags-per-build", "1", "--workers", "2",
                     "--prompt", "1", "--generate", "1", "--repetitions", "1",
                     "--build-dir", str(base / "build"),
                     "--results-dir", str(base / "sweep-results")]
            subprocess.run(sweep, check=True, capture_output=True, text=True)
            planned = json.loads((base / "sweep-results/subset-plan.json").read_text())
            executed = json.loads((base / "sweep-results/subset-executions.json").read_text())
            self.assertEqual(len(planned), len(executed))
            self.assertEqual(executed[0]["status"], "completed")
            self.assertEqual(executed[0]["config"], planned[0]["config"])
            self.assertEqual(json.loads((base / "sweep-results/subset-settings.json").read_text())
                             ["gpu_layers"], 0)
            self.assertGreater(len(executed[0]["config"]["cmake"]), 10)
            subprocess.run(sweep, check=True, capture_output=True, text=True)
            self.assertEqual(len(json.loads((base / "sweep-results/subset-executions.json").read_text())), 1)
            failure_command = [*sweep[:-2], "--results-dir", str(base / "failure-results")]
            failed = subprocess.run(failure_command, capture_output=True, text=True,
                                    env={**os.environ, "LLAMA_BENCH_TEST_FAIL": "1"})
            self.assertEqual(failed.returncode, 1)
            self.assertIn("1 failed", failed.stdout)
            record_path = base / "failure-results/subset-executions.json"
            previous = json.loads(record_path.read_text())[0]
            self.assertEqual(previous["status"], "failed")
            self.assertEqual(previous["exit_code"], 1)
            retried = subprocess.run(failure_command, check=True, capture_output=True, text=True)
            self.assertIn("Retrying failed execution", retried.stdout)
            current = json.loads(record_path.read_text())[0]
            self.assertEqual(current["status"], "completed")
            self.assertEqual(current["previous_attempts"][0]["results_directory"],
                             previous["results_directory"])


if __name__ == "__main__":
    unittest.main()
