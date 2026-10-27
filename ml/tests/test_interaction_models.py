"""Regression checks for learned flag combinations and their CLI integration."""

import contextlib
import io
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.evaluate import leave_one_platform_out
from ml.main import main, model_path
from ml.models import (BaseModel, InteractionPerformanceModel,
                       NoHardwareInteractionModel, NoHardwareModel, PerformanceModel)
from ml.models.common import configs


def measurements():
    # A hurts alone, but helps when B is enabled. The additive baseline chooses
    # B alone; the interaction model should recover the superior A+B build.
    effects = {("OFF", "OFF"): -.1, ("ON", "OFF"): -1.5,
               ("OFF", "ON"): -.2, ("ON", "ON"): 0.0}
    return pd.DataFrame([
        {"application": "llama.cpp", "platform": str(machine), "hw.cores": machine + 1,
         "cfg.GGML_LTO": a, "cfg.GGML_OPENMP": b, "runs": 1,
         "relative": np.exp(effects[a, b])}
        for machine in range(12)
        for a, b in itertools.product(["OFF", "ON"], repeat=2)
    ])


class InteractionModelTests(unittest.TestCase):
    def setUp(self):
        self.table = measurements()
        self.hardware = {"cores": 5}
        self.builds = configs(self.table.iloc[:4])

    def test_learns_conditional_effect_and_improves_fit(self):
        original = PerformanceModel().fit(self.table)
        model = InteractionPerformanceModel().fit(self.table)
        old, _ = original.predict_performance(self.hardware, self.builds)
        new, std = model.predict_performance(self.hardware, self.builds)
        measured = self.table["relative"].iloc[:4].to_numpy()
        self.assertLess(np.mean(abs(new - measured)), .01)
        self.assertGreater(np.mean(abs(old - measured)), .1)
        # Row order: neither, B, A, A+B. A reverses sign depending on B.
        self.assertLess(new[2], new[0])
        self.assertGreater(new[3], new[1])
        self.assertTrue(np.isfinite(std).all())
        self.assertTrue((std >= 0).all())
        self.assertFalse(hasattr(original, "option_pairs"))

    def test_no_hardware_baseline_selects_joint_optimum(self):
        model = NoHardwareInteractionModel().fit(self.table)
        best, _ = model.predict(self.hardware)
        self.assertEqual(best, {"GGML_LTO": "ON", "GGML_OPENMP": "ON"})
        additive, _ = NoHardwareModel().fit(self.table).predict(self.hardware)
        self.assertEqual(additive, {"GGML_LTO": "OFF", "GGML_OPENMP": "ON"})
        self.assertEqual(model.hw_features, [])
        first = model.predict_performance({"cores": 1}, self.builds)
        second = model.predict_performance({"cores": 1000}, self.builds)
        np.testing.assert_array_equal(first, second)

    def test_retains_hardware_terms_but_no_three_way_terms(self):
        model = InteractionPerformanceModel().fit(self.table)
        X = model._encode(pd.DataFrame([{**self.hardware, **c} for c in self.builds]))
        self.assertIn("GGML_LTO_ON x cores", X)
        self.assertIn("GGML_LTO_ON x GGML_OPENMP_ON", X)
        self.assertTrue(all(column.count(" x ") <= 1 for column in X))

    def test_pair_schema_is_stable_for_single_build_and_after_reload(self):
        for cls in [InteractionPerformanceModel, NoHardwareInteractionModel]:
            with self.subTest(model=cls.__name__), tempfile.TemporaryDirectory() as tmp:
                model = cls().fit(self.table)
                mean, spread = model.predict_performance(self.hardware, self.builds)
                for i, build in enumerate(self.builds):
                    single = model.predict_performance(self.hardware, [build])
                    np.testing.assert_allclose([single[0][0], single[1][0]], [mean[i], spread[i]])
                path = Path(tmp) / "model.joblib"
                model.save(path)
                restored = BaseModel.load(path)
                np.testing.assert_allclose(restored.predict_performance(self.hardware, self.builds),
                                           (mean, spread))
                self.assertEqual(restored.option_pairs, model.option_pairs)

    def test_same_option_and_constant_products_are_excluded(self):
        table = self.table.copy()
        table["cfg.GGML_BLAS_VENDOR"] = ["Generic", "OpenBLAS", "Apple", "Generic"] * 12
        model = InteractionPerformanceModel().fit(table)
        self.assertFalse(any(a.startswith("GGML_BLAS_VENDOR_") and b.startswith("GGML_BLAS_VENDOR_")
                             for a, b in model.option_pairs))
        # A and B never enabled together: their product contains no evidence.
        subset = self.table[~((self.table["cfg.GGML_LTO"] == "ON") &
                              (self.table["cfg.GGML_OPENMP"] == "ON"))]
        model.fit(subset)
        self.assertEqual(model.option_pairs, [])

    def test_boolean_options_are_encoded_as_options(self):
        table = self.table.copy()
        for column in ["cfg.GGML_LTO", "cfg.GGML_OPENMP"]:
            table[column] = table[column] == "ON"
        model = InteractionPerformanceModel().fit(table)
        self.assertEqual(model.option_pairs, [("GGML_LTO_True", "GGML_OPENMP_True")])
        self.assertNotIn("GGML_LTO", model.hw_columns)

    def test_constant_options_have_no_pairs(self):
        table = pd.concat([self.table.iloc[[0]]] * 2, ignore_index=True)
        model = InteractionPerformanceModel().fit(table)
        self.assertEqual(model.option_pairs, [])
        self.assertTrue(np.isfinite(model.predict(self.hardware)[1]))

    def test_suggestions_and_decision_costs_use_pair_effects(self):
        model = NoHardwareInteractionModel().fit(self.table)
        suggestions = model.suggest(self.hardware, [], n=1)
        self.assertEqual(suggestions[0][0], {"GGML_LTO": "ON", "GGML_OPENMP": "ON"})
        best = model.predict(self.hardware)[1]
        decisions = model.decisions(self.hardware)
        for _, decision in decisions.iterrows():
            variant = {"GGML_LTO": "ON", "GGML_OPENMP": "ON", decision["option"]: decision["alternative"]}
            predicted = model.predict_performance(self.hardware, [variant])[0][0]
            self.assertAlmostEqual(decision["cost"], predicted / best - 1)

    def test_evaluation_compares_both_interaction_variants(self):
        result = leave_one_platform_out(self.table, random_trials=5)
        self.assertTrue((result["interactions"] == 1.0).all())
        self.assertTrue((result["no_hardware_interactions"] == 1.0).all())
        self.assertTrue((result["error_interactions"] < .01).all())
        self.assertTrue((result["no_hardware"] < 1.0).all())

    def test_empty_evaluation_fold_has_missing_interaction_scores(self):
        table = self.table.iloc[:8].copy()
        table.loc[table["platform"] == "0", "cfg.GGML_LTO"] = "unsupported"
        result = leave_one_platform_out(table, random_trials=1)
        self.assertTrue((result["runs"] == 0).all())
        self.assertTrue(result[["interactions", "error_interactions",
                                "no_hardware_interactions"]].isna().all().all())

    def test_gromacs_options_use_the_same_pair_encoder(self):
        table = self.table.rename(columns={"cfg.GGML_LTO": "cfg.GMX_HWLOC",
                                          "cfg.GGML_OPENMP": "cfg.GMX_OPENMP"}).copy()
        table["application"] = "gromacs"
        for cls in [InteractionPerformanceModel, NoHardwareInteractionModel]:
            with self.subTest(model=cls.__name__):
                model = cls().fit(table)
                best, _ = model.predict(self.hardware)
                self.assertEqual(best, {"GMX_HWLOC": "ON", "GMX_OPENMP": "ON"})

    def test_no_hardware_keeps_hardware_derived_options_and_validity(self):
        table = self.table.copy()
        table["cfg.CMAKE_CUDA_ARCHITECTURES"] = "80"
        table["cfg.CMAKE_C_COMPILER"] = "gcc"
        model = NoHardwareInteractionModel().fit(table)
        self.assertNotIn("CMAKE_CUDA_ARCHITECTURES", model.options)
        first, _ = model.predict({"n_gpus": 1, "gpu_compute_capability": 8.0})
        second, _ = model.predict({"n_gpus": 1, "gpu_compute_capability": 9.0})
        self.assertEqual(first["CMAKE_CUDA_ARCHITECTURES"], "80")
        self.assertEqual(second["CMAKE_CUDA_ARCHITECTURES"], "90")
        # If every available build requires LTO, a compiler lacking LTO rejects all.
        model.values["GGML_LTO"] = ["ON"]
        with self.assertRaisesRegex(ValueError, "no valid build"):
            model.predict({"n_gpus": 0, "gpu_compute_capability": 0, "gcc_lto": False,
                           "gcc_openmp": False})

    def test_cli_trains_and_saves_both_variants(self):
        for kind, cls in [("hardware-interactions", InteractionPerformanceModel),
                          ("no-hardware-interactions", NoHardwareInteractionModel)]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                args = ["ml.main", "--models", tmp, "train", "--model", kind]
                with patch("sys.argv", args), patch("ml.main.load_table", return_value=self.table), \
                        contextlib.redirect_stdout(io.StringIO()):
                    main()
                self.assertIsInstance(BaseModel.load(model_path(tmp, "llama.cpp", kind)), cls)

                # Use the same saved variant through the prediction command.
                args = ["ml.main", "--models", tmp, "predict", "--model", kind,
                        "--system", "unused.json", "--app", "llama.cpp"]
                output = io.StringIO()
                with patch("sys.argv", args), \
                        patch("ml.main.read_hardware", return_value=("target", self.hardware)), \
                        contextlib.redirect_stdout(output):
                    main()
                self.assertIn("GGML_LTO=ON", output.getvalue())

    def test_cli_suggestions_use_selected_interaction_model(self):
        for kind in ["hardware-interactions", "no-hardware-interactions"]:
            with self.subTest(kind=kind):
                args = ["ml.main", "suggest", "--model", kind, "--system", "unused.json", "-n", "1"]
                output = io.StringIO()
                with patch("sys.argv", args), patch("ml.main.load_table", return_value=self.table), \
                        patch("ml.main.read_hardware", return_value=("target", self.hardware)), \
                        contextlib.redirect_stdout(output):
                    main()
                self.assertIn("GGML_LTO=ON", output.getvalue())
                self.assertIn("GGML_OPENMP=ON", output.getvalue())


if __name__ == "__main__":
    unittest.main()
