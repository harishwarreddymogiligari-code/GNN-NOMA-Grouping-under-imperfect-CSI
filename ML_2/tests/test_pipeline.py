import unittest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import pandas as pd
import numpy as np
import os
import shutil

import config
from data.generate_demo_data import generate_noma_dataset
from preprocessing import load_raw_dataset, FeatureScaler, build_pyg_datasets
from models import UncertaintyAwareGNN
from training import SupervisedLoss, GraphPerturbedConsistencyLoss, ConfidenceUncertaintyPseudoLabeler, SemiSupervisedTrainer


class TestNOMAGNNPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_scratch"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.csv_path = os.path.join(cls.test_dir, "test_dataset.csv")

        # Generate small test dataset (5 scenarios, 10 users each)
        df = generate_noma_dataset(num_scenarios=5, users_per_scenario=10, num_groups=4, seed=123)
        df.to_csv(cls.csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_data_loader(self):
        df = load_raw_dataset(self.csv_path)
        self.assertIn("csi_uncertainty", df.columns)
        self.assertEqual(len(df), 50)

    def test_csi_uncertainty_derivation(self):
        df = load_raw_dataset(self.csv_path)
        df_no_uncert = df.drop(columns=["csi_uncertainty", "csi_error"])
        temp_csv = os.path.join(self.test_dir, "temp_no_uncert.csv")
        df_no_uncert.to_csv(temp_csv, index=False)

        df_derived = load_raw_dataset(temp_csv)
        self.assertIn("csi_uncertainty", df_derived.columns)
        self.assertIn("csi_error", df_derived.columns)

    def test_feature_scaler(self):
        df = load_raw_dataset(self.csv_path)
        scaler = FeatureScaler(config.FEATURE_COLUMNS)
        scaler.fit(df)
        df_scaled = scaler.transform(df)
        self.assertAlmostEqual(df_scaled[config.FEATURE_COLUMNS[0]].mean(), 0.0, places=4)

    def test_graph_builder(self):
        df = load_raw_dataset(self.csv_path)
        scaler = FeatureScaler(config.FEATURE_COLUMNS)
        df_scaled = scaler.fit_transform(df)

        graphs = build_pyg_datasets(df_scaled, config.FEATURE_COLUMNS, config.LABEL_COLUMN, method="knn", k_neighbors=3)
        self.assertEqual(len(graphs), 5)
        self.assertEqual(graphs[0].num_nodes, 10)
        self.assertEqual(graphs[0].x.shape[1], len(config.FEATURE_COLUMNS))

    def test_gnn_model_forward(self):
        df = load_raw_dataset(self.csv_path)
        scaler = FeatureScaler(config.FEATURE_COLUMNS)
        df_scaled = scaler.fit_transform(df)
        graphs = build_pyg_datasets(df_scaled, config.FEATURE_COLUMNS, config.LABEL_COLUMN, method="knn", k_neighbors=3)

        model = UncertaintyAwareGNN(in_channels=len(config.FEATURE_COLUMNS), hidden_dim=16, out_channels=4, gnn_type="GraphSAGE")
        g0 = graphs[0]
        probs, embeddings, logits = model(g0.x, g0.edge_index, g0.edge_attr)

        self.assertEqual(probs.shape, (10, 4))
        self.assertEqual(embeddings.shape, (10, 16))
        self.assertAlmostEqual(probs.sum(dim=-1).mean().item(), 1.0, places=4)

    def test_semi_supervised_losses(self):
        df = load_raw_dataset(self.csv_path)
        scaler = FeatureScaler(config.FEATURE_COLUMNS)
        df_scaled = scaler.fit_transform(df)
        graphs = build_pyg_datasets(df_scaled, config.FEATURE_COLUMNS, config.LABEL_COLUMN, method="knn", k_neighbors=3)
        g0 = graphs[0]

        model = UncertaintyAwareGNN(in_channels=len(config.FEATURE_COLUMNS), hidden_dim=16, out_channels=4)
        probs, embeddings, logits = model(g0.x, g0.edge_index, g0.edge_attr)

        # Mask half as unlabeled (-1)
        g0.y[5:] = -1

        sup_loss_fn = SupervisedLoss()
        cons_loss_fn = GraphPerturbedConsistencyLoss()
        pseudo_loss_fn = ConfidenceUncertaintyPseudoLabeler(confidence_threshold=0.5)

        loss_sup = sup_loss_fn(logits, g0.y)
        loss_cons = cons_loss_fn(model, g0.x, g0.edge_index, g0.edge_attr, (g0.y == -1))
        loss_pseudo, stats = pseudo_loss_fn(logits, probs, g0.csi_uncertainty, (g0.y == -1))

        self.assertGreaterEqual(loss_sup.item(), 0.0)
        self.assertGreaterEqual(loss_cons.item(), 0.0)
        self.assertGreaterEqual(loss_pseudo.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
