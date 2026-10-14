import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

import config
from .supervised_loss import SupervisedLoss
from .consistency_loss import GraphPerturbedConsistencyLoss
from .pseudo_labeling import ConfidenceUncertaintyPseudoLabeler
from evaluation.metrics import calculate_classification_metrics


class SemiSupervisedTrainer:
    """
    Semi-Supervised Trainer for Uncertainty-Aware GNN NOMA User Grouping.
    
    Manages end-to-end training using the composite loss:
    Total Loss = Supervised Loss + lambda * Consistency Loss + mu * Pseudo-Label Loss
    """
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        lambda_consistency: float = config.LAMBDA_CONSISTENCY,
        mu_pseudo: float = config.MU_PSEUDO,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        uncertainty_threshold: float = config.UNCERTAINTY_THRESHOLD,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_consistency = lambda_consistency
        self.mu_pseudo = mu_pseudo

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Loss functions
        self.supervised_criterion = SupervisedLoss()
        self.consistency_criterion = GraphPerturbedConsistencyLoss(
            edge_drop_rate=config.PERTURB_EDGE_DROP_RATE,
            feature_mask_rate=config.PERTURB_FEATURE_MASK_RATE
        )
        self.pseudo_labeler = ConfidenceUncertaintyPseudoLabeler(
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold
        )

        self.history = {
            "epoch": [],
            "total_loss": [],
            "supervised_loss": [],
            "consistency_loss": [],
            "pseudo_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "pseudo_coverage": []
        }

    def train_epoch(self, train_graphs: List[torch.nn.Module]) -> Dict[str, float]:
        """
        Executes one epoch of semi-supervised GNN training over the training graphs.
        """
        self.model.train()
        total_loss_sum = 0.0
        sup_loss_sum = 0.0
        cons_loss_sum = 0.0
        pseudo_loss_sum = 0.0
        coverage_sum = 0.0
        total_nodes = 0

        for graph in train_graphs:
            graph = graph.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass on original graph
            probs, embeddings, logits = self.model(graph.x, graph.edge_index, graph.edge_attr)

            # Node masks
            labeled_mask = (graph.y != -1)
            unlabeled_mask = (graph.y == -1)

            # 1. Supervised Loss on labeled nodes
            loss_sup = self.supervised_criterion(logits, graph.y)

            # 2. Dual-Perturbation Consistency Loss on unlabeled nodes
            loss_cons = self.consistency_criterion(
                self.model, graph.x, graph.edge_index, graph.edge_attr, unlabeled_mask
            )

            # 3. Confidence & CSI Uncertainty Pseudo-labeling Loss
            loss_pseudo, pseudo_stats = self.pseudo_labeler(
                logits, probs, graph.csi_uncertainty, unlabeled_mask
            )

            # Total Composite Loss
            loss_total = loss_sup + (self.lambda_consistency * loss_cons) + (self.mu_pseudo * loss_pseudo)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss_sum += loss_total.item() * graph.num_nodes
            sup_loss_sum += loss_sup.item() * graph.num_nodes
            cons_loss_sum += loss_cons.item() * graph.num_nodes
            pseudo_loss_sum += loss_pseudo.item() * graph.num_nodes
            coverage_sum += pseudo_stats.get("coverage", 0.0) * graph.num_nodes
            total_nodes += graph.num_nodes

        avg_total_loss = total_loss_sum / total_nodes if total_nodes > 0 else 0.0
        avg_sup_loss = sup_loss_sum / total_nodes if total_nodes > 0 else 0.0
        avg_cons_loss = cons_loss_sum / total_nodes if total_nodes > 0 else 0.0
        avg_pseudo_loss = pseudo_loss_sum / total_nodes if total_nodes > 0 else 0.0
        avg_coverage = coverage_sum / total_nodes if total_nodes > 0 else 0.0

        return {
            "total_loss": avg_total_loss,
            "supervised_loss": avg_sup_loss,
            "consistency_loss": avg_cons_loss,
            "pseudo_loss": avg_pseudo_loss,
            "pseudo_coverage": avg_coverage
        }

    @torch.no_grad()
    def evaluate(self, val_graphs: List[torch.nn.Module]) -> Dict[str, float]:
        """
        Evaluates current GNN model performance on validation/test graphs.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        val_loss_sum = 0.0
        total_nodes = 0

        for graph in val_graphs:
            graph = graph.to(self.device)
            probs, _, logits = self.model(graph.x, graph.edge_index, graph.edge_attr)

            labeled_mask = (graph.y != -1)
            if labeled_mask.any():
                loss = F.cross_entropy(logits[labeled_mask], graph.y[labeled_mask])
                val_loss_sum += loss.item() * labeled_mask.sum().item()
                total_nodes += labeled_mask.sum().item()

                preds = torch.argmax(probs[labeled_mask], dim=-1).cpu().numpy()
                targets = graph.y[labeled_mask].cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_val_loss = val_loss_sum / total_nodes if total_nodes > 0 else 0.0

        if len(all_preds) > 0:
            metrics = calculate_classification_metrics(all_targets, all_preds)
            metrics["val_loss"] = avg_val_loss
        else:
            metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "val_loss": avg_val_loss}

        return metrics

    def fit(
        self,
        train_graphs: List[torch.nn.Module],
        val_graphs: List[torch.nn.Module],
        epochs: int = config.EPOCHS,
        model_save_dir: str = config.MODEL_SAVE_DIR
    ) -> Dict[str, List[float]]:
        """
        Executes full semi-supervised training procedure with early checkpointing.
        """
        os.makedirs(model_save_dir, exist_ok=True)
        best_val_f1 = -1.0
        best_model_path = os.path.join(model_save_dir, "best_semi_supervised_gnn.pt")
        final_model_path = os.path.join(model_save_dir, "final_semi_supervised_gnn.pt")

        print(f"\n[TRAINER] Starting Semi-Supervised GNN Training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_graphs)
            val_metrics = self.evaluate(val_graphs)

            self.history["epoch"].append(epoch)
            self.history["total_loss"].append(train_metrics["total_loss"])
            self.history["supervised_loss"].append(train_metrics["supervised_loss"])
            self.history["consistency_loss"].append(train_metrics["consistency_loss"])
            self.history["pseudo_loss"].append(train_metrics["pseudo_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_score"])
            self.history["pseudo_coverage"].append(train_metrics["pseudo_coverage"])

            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Loss: {train_metrics['total_loss']:.4f} (Sup: {train_metrics['supervised_loss']:.4f}, "
                    f"Cons: {train_metrics['consistency_loss']:.4f}, Ps: {train_metrics['pseudo_loss']:.4f}) | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1_score']:.4f}"
                )

            # Checkpoint best model state based on validation F1 score
            if val_metrics["f1_score"] > best_val_f1:
                best_val_f1 = val_metrics["f1_score"]
                torch.save(self.model.state_dict(), best_model_path)

        # Save final model state
        torch.save(self.model.state_dict(), final_model_path)
        
        # Load best model checkpoint back into self.model for downstream evaluation
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        print(f"[TRAINER] Training Complete! Best Val F1: {best_val_f1:.4f}. Restored best checkpoint: {best_model_path}")

        # Save training history CSV
        os.makedirs(os.path.join(config.RESULTS_DIR, "predictions"), exist_ok=True)
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(config.RESULTS_DIR, "predictions", "training_history.csv"), index=False)

        return self.history
