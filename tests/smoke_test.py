"""Lightweight smoke tests for Beluga ForecastNet."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

GEFCOM_PATH = PROJECT_ROOT / "dataset" / "GEFCom2014 Data"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class BelugaForecastNetSmokeTests(unittest.TestCase):
    @unittest.skipUnless(has_module("numpy"), "numpy is not installed")
    def test_bwo_runs_with_dummy_fitness(self) -> None:
        from lfs_hdlbwo.optimization import BWO

        optimizer = BWO(
            population_size=3,
            max_iter=2,
            dimension=2,
            lower_bound=[0, 0],
            upper_bound=[1, 1],
            seed=1,
            verbose=False,
        )
        position, score = optimizer.optimize(lambda x: float((x**2).sum()))
        self.assertEqual(position.shape, (2,))
        self.assertGreaterEqual(score, 0.0)

    @unittest.skipUnless(has_module("numpy") and has_module("pandas"), "numpy and pandas are not installed")
    def test_windowing_shape(self) -> None:
        import pandas as pd

        from comparison_models.robust_features import create_raw_windows

        frame = pd.DataFrame({"load": range(20), "temperature": range(20)})
        x, y = create_raw_windows(frame, window_size=14, forecast_horizon=1, target_feature="load")
        self.assertEqual(x.shape, (6, 14, 2))
        self.assertEqual(y.shape, (6,))

    @unittest.skipUnless(has_module("tensorflow"), "tensorflow is not installed")
    def test_cblstm_ae_builds(self) -> None:
        from lfs_hdlbwo.model import build_cblstm_ae

        model = build_cblstm_ae(sequence_length=14, num_features=8)
        self.assertEqual(model.output_shape, (None, 14, 1))

    @unittest.skipUnless(has_module("tensorflow"), "tensorflow is not installed")
    def test_baseline_models_build(self) -> None:
        from comparison_models.deep_learning import (
            build_bilstm_baseline,
            build_cblstm_ae_baseline,
            build_cnn_bilstm_baseline,
            build_cnn_lstm_baseline,
            build_gru_baseline,
            build_lstm_baseline,
        )

        for builder in (
            build_lstm_baseline,
            build_gru_baseline,
            build_bilstm_baseline,
            build_cnn_lstm_baseline,
            build_cnn_bilstm_baseline,
        ):
            model = builder(sequence_length=14, num_features=8)
            self.assertEqual(model.output_shape, (None, 1))

        autoencoder = build_cblstm_ae_baseline(sequence_length=14, num_features=8)
        self.assertEqual(autoencoder.output_shape, (None, 14, 1))

    @unittest.skipUnless(
        has_module("numpy") and has_module("pandas") and GEFCOM_PATH.exists(),
        "numpy, pandas, and GEFCom2014 dataset are required",
    )
    def test_gefcom_adapter_default_slice(self) -> None:
        import numpy as np

        from lfs_hdlbwo.gefcom_dataset import build_gefcom_load_features

        features = build_gefcom_load_features(GEFCOM_PATH)
        metadata = features.attrs["metadata"]
        self.assertEqual(metadata["selected_rows"], 8760)
        self.assertEqual(metadata["target_series_names"], ["load"])
        self.assertEqual(metadata["exogenous_series_names"], ["temperature"])
        self.assertIn("load_lag_168", features.columns)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    @unittest.skipUnless(
        has_module("numpy") and has_module("pandas") and GEFCOM_PATH.exists(),
        "numpy, pandas, and GEFCom2014 dataset are required",
    )
    def test_prepare_small_range(self) -> None:
        from comparison_models.robust_features import FEATURE_MODE_GEFCOM_LOAD, prepare_robust_dataset

        prepared = prepare_robust_dataset(
            dataset_path=GEFCOM_PATH,
            feature_mode=FEATURE_MODE_GEFCOM_LOAD,
            window_size=24,
            validation_split=0.2,
            start_date="2014-01-01",
            end_date="2014-01-31",
        )
        self.assertEqual(prepared.x_train.shape[1], 24)
        self.assertEqual(prepared.x_train.shape[2], len(prepared.feature_names))
        self.assertEqual(prepared.metadata["selected_rows"], 744)
        self.assertGreater(len(prepared.x_train), len(prepared.x_val))

    @unittest.skipUnless(has_module("numpy") and has_module("tensorflow"), "numpy and tensorflow are not installed")
    def test_bwo_decode(self) -> None:
        import numpy as np

        from lfs_hdlbwo.proposed import decode_proposed_position

        params = decode_proposed_position(np.zeros(6))
        self.assertEqual(params.conv_filters, 32)
        self.assertEqual(params.batch_size, 32)
        self.assertEqual(params.dense_units, 16)


if __name__ == "__main__":
    unittest.main()
