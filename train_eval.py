#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenamiento y evaluación de modelos para predicción de fallos en casos de prueba.
Lee dataset JSON, entrena baselines (sklearn) y un MLP (TensorFlow), optimiza umbral por F1,
genera métricas, gráficas y un reporte Markdown.

Uso:
  python train_eval.py --data dataset_pruebas_sintetico.json --out ./out --export-weka --plots --calibration

Requisitos:
  - numpy, pandas, scikit-learn, imbalanced-learn, tensorflow>=2.14,<3, matplotlib
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# =========================
# Helpers de logging/plots
# =========================
def print_header(msg):
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))

def buscar_umbral_mejor_f1(prob, y_true):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def plot_roc(y_true, prob, title, path_png):
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()
    return float(roc_auc)

def plot_pr(y_true, prob, title, path_png):
    precision, recall, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()
    return float(ap)

def plot_cm(y_true, y_pred, title, path_png):
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()
    return cm

def plot_calibration(y_true, prob, title, path_png, n_bins=10):
    fig, ax = plt.subplots()
    CalibrationDisplay.from_predictions(y_true, prob, n_bins=n_bins, strategy="quantile", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

def evaluar_sklearn_kfold(modelo, X_raw, y, preprocessor, k=5, threshold=0.5, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs, precs, recs, f1s, aucs = [], [], [], [], []
    for tr, te in skf.split(X_raw, y):
        pipe = Pipeline([("pre", preprocessor), ("clf", modelo)])
        pipe.fit(X_raw.iloc[tr], y[tr])

        if hasattr(pipe["clf"], "predict_proba"):
            prob = pipe.predict_proba(X_raw.iloc[te])[:, 1]
        elif hasattr(pipe["clf"], "decision_function"):
            scores = pipe["clf"].decision_function(X_raw.iloc[te])
            prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            pred_hard = pipe.predict(X_raw.iloc[te])
            prob = pred_hard.astype(float)

        pred = (prob >= threshold).astype(int)

        accs.append(accuracy_score(y[te], pred))
        p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average="binary", zero_division=0)
        precs.append(p); recs.append(r); f1s.append(f1)
        try:
            aucs.append(roc_auc_score(y[te], prob))
        except Exception:
            aucs.append(np.nan)

    return {
        "Accuracy": float(np.nanmean(accs)),
        "Precision": float(np.nanmean(precs)),
        "Recall": float(np.nanmean(recs)),
        "F1": float(np.nanmean(f1s)),
        "AUC": float(np.nanmean(aucs)),
    }

def build_mlp(input_dim: int):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluación de modelos para QA predictivo (v2).")
    parser.add_argument("--data", type=str, required=True, help="Ruta al JSON del dataset.")
    parser.add_argument("--out", type=str, default="./out", help="Carpeta de salida.")
    parser.add_argument("--k", type=int, default=5, help="K folds para baselines sklearn.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Porción test para MLP/baselines (figuras).")
    parser.add_argument("--epochs", type=int, default=200, help="Épocas para MLP.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size para MLP.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--export-weka", action="store_true", help="Exportar CSV listo para WEKA.")
    parser.add_argument("--plots", action="store_true", help="Genera y guarda ROC/PR/CM por modelo.")
    parser.add_argument("--calibration", action="store_true", help="Genera curva de calibración por modelo.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    figs_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plots or args.calibration:
        figs_dir.mkdir(parents=True, exist_ok=True)

    # Semillas
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ----------------- Cargar datos -----------------
    json_path = Path(args.data)
    assert json_path.exists(), f"No existe: {json_path}"
    df = pd.read_json(json_path, encoding="utf-8")

    # Etiqueta binaria (Fallida=1)
    df["target"] = (df["result"].str.lower() == "fallida").astype(int)
    cat_cols = ["type","methodology","phase","origin","complexity","priority"]

    # Evitar fuga de información: no usar p_fail_model como feature
    X_raw = df[cat_cols].copy()
    y = df["target"].values

    # Preprocesamiento (One-Hot denso)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer([("cat", ohe, cat_cols)], remainder="drop")

    # ================= Baselines (k-fold) =================
    print_header("BASelines (sklearn, k-fold)")
    baselines = {
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=args.seed),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=args.seed, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=args.seed, class_weight="balanced"),
    }
    baseline_kfold = {}
    for name, clf in baselines.items():
        m = evaluar_sklearn_kfold(clf, X_raw, y, preprocessor=pre, k=args.k, threshold=0.5, seed=args.seed)
        baseline_kfold[name] = m
        print(f"{name:18s} | Acc={m['Accuracy']:.3f}  P={m['Precision']:.3f}  R={m['Recall']:.3f}  F1={m['F1']:.3f}  AUC={m['AUC']:.3f}")

    (out_dir / "baselines_kfold_results.json").write_text(json.dumps(baseline_kfold, indent=2))

    # ===== Split fijo para comparaciones y gráficas =====
    X_tr_idx, X_te_idx, y_tr_idx, y_te_idx = train_test_split(
        X_raw.index, df["target"].values, test_size=args.test_size,
        stratify=df["target"].values, random_state=args.seed
    )
    y_tr = df.loc[X_tr_idx, "target"].values
    y_te = df.loc[X_te_idx, "target"].values

    # Entrenamiento y gráficos por baseline en split fijo
    print_header("Baselines (split fijo) para métricas y figuras")
    fixed_split_metrics = {}

    def entrenar_y_reportar_baseline(name, clf):
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_raw.loc[X_tr_idx], y_tr)

        # Probas
        if hasattr(pipe["clf"], "predict_proba"):
            prob = pipe.predict_proba(X_raw.loc[X_te_idx])[:, 1]
        elif hasattr(pipe["clf"], "decision_function"):
            scores = pipe["clf"].decision_function(X_raw.loc[X_te_idx])
            prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            prob = pipe.predict(X_raw.loc[X_te_idx]).astype(float)

        # Umbral óptimo por F1
        tau_b, _ = buscar_umbral_mejor_f1(prob, y_te)
        pred = (prob >= tau_b).astype(int)

        # Métricas
        acc = accuracy_score(y_te, pred)
        p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
        auc_roc = roc_auc_score(y_te, prob)
        ap = average_precision_score(y_te, prob)

        # Plots
        if args.plots:
            plot_roc(y_te, prob, f"ROC - {name}", figs_dir / f"roc_{name}.png")
            plot_pr(y_te, prob, f"PR - {name}", figs_dir / f"pr_{name}.png")
            plot_cm(y_te, pred, f"CM (τ={tau_b:.2f}) - {name}", figs_dir / f"cm_{name}.png")
        if args.calibration:
            plot_calibration(y_te, prob, f"Calibration - {name}", figs_dir / f"cal_{name}.png")

        fixed_split_metrics[name] = {
            "Accuracy": float(acc),
            "Precision": float(p),
            "Recall": float(r),
            "F1": float(f1),
            "ROC_AUC": float(auc_roc),
            "PR_AP": float(ap),
            "BestThresholdF1": float(tau_b)
        }

    for name, clf in baselines.items():
        entrenar_y_reportar_baseline(name, clf)

    # ================== MLP (TensorFlow) ==================
    print_header("MLP (TensorFlow)")
    X_mat = pre.fit_transform(X_raw)
    input_dim = X_mat.shape[1]
    X_tr, X_te, y_tr_tf, y_te_tf = train_test_split(
        X_mat, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    cw = class_weight.compute_class_weight(class_weight="balanced",
                                           classes=np.unique(y_tr_tf),
                                           y=y_tr_tf)
    class_weights = {0: cw[0], 1: cw[1]}

    model = build_mlp(input_dim)
    cb_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_tr, y_tr_tf,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        class_weight=class_weights,
        callbacks=[cb_early]
    )

    prob_te = model.predict(X_te, verbose=0).ravel()
    tau, best_f1 = buscar_umbral_mejor_f1(prob_te, y_te_tf)
    pred_te = (prob_te >= tau).astype(int)

    acc = accuracy_score(y_te_tf, pred_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te_tf, pred_te, average="binary", zero_division=0)
    auc_roc = roc_auc_score(y_te_tf, prob_te)
    ap = average_precision_score(y_te_tf, prob_te)

    mlp_report = {
        "Accuracy": float(acc),
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        "ROC_AUC": float(auc_roc),
        "PR_AP": float(ap),
        "BestThresholdF1": float(tau),
        "BestF1_on_val": float(best_f1)
    }
    print(f"MLP(TF)       | Acc={acc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  ROC_AUC={auc_roc:.3f}  PR_AP={ap:.3f}  (τ*={tau:.2f})")
    print("\nMatriz de confusión (test):")
    print(confusion_matrix(y_te_tf, pred_te))
    print("\nReporte de clasificación (test):")
    print(classification_report(y_te_tf, pred_te, digits=3))

    # Gráficos MLP
    if args.plots:
        plot_roc(y_te_tf, prob_te, "ROC - MLP", figs_dir / "roc_MLP.png")
        plot_pr(y_te_tf, prob_te, "PR - MLP", figs_dir / "pr_MLP.png")
        plot_cm(y_te_tf, pred_te, f"CM (τ={tau:.2f}) - MLP", figs_dir / "cm_MLP.png")
    if args.calibration:
        plot_calibration(y_te_tf, prob_te, "Calibration - MLP", figs_dir / "cal_MLP.png")

    # Guardar resultados
    (out_dir / "baselines_fixedsplit_metrics.json").write_text(json.dumps(fixed_split_metrics, indent=2))
    (out_dir / "mlp_results.json").write_text(json.dumps(mlp_report, indent=2))
    model.save(out_dir / "mlp_model.h5")

    # Export WEKA
    if args.export_weka:
        weka_df = df[["type","methodology","phase","origin","complexity","priority","result"]].copy()
        out_weka = out_dir / "dataset_pruebas_para_weka.csv"
        weka_df.to_csv(out_weka, index=False, encoding="utf-8")
        print(f"\n[OK] Exportado para WEKA: {out_weka.resolve()}")

    # ================== Reporte Markdown ==================
    print_header("Generando report.md")
    report_path = out_dir / "report.md"
    lines = []
    lines.append(f"# Reporte de modelos (QA predictivo)\n")
    lines.append(f"- Dataset: `{json_path.name}`")
    lines.append(f"- Test size: {args.test_size}, Seed: {args.seed}\n")
    lines.append("## Métricas (split fijo)\n")

    def block_metrics(name, m):
        return (f"**{name}**\n\n"
                f"- Accuracy: {m['Accuracy']:.3f}\n"
                f"- Precision: {m['Precision']:.3f}\n"
                f"- Recall: {m['Recall']:.3f}\n"
                f"- F1: {m['F1']:.3f}\n"
                f"- ROC AUC: {m['ROC_AUC']:.3f}\n"
                f"- PR AP: {m['PR_AP']:.3f}\n"
                f"- Umbral F1*: {m['BestThresholdF1']:.2f}\n")

    for name in ["LogisticRegression","DecisionTree","RandomForest"]:
        lines.append(block_metrics(name, fixed_split_metrics[name]) + "\n")

    lines.append("**MLP (TensorFlow)**\n")
    lines.append(f"- Accuracy: {mlp_report['Accuracy']:.3f}\n"
                 f"- Precision: {mlp_report['Precision']:.3f}\n"
                 f"- Recall: {mlp_report['Recall']:.3f}\n"
                 f"- F1: {mlp_report['F1']:.3f}\n"
                 f"- ROC AUC: {mlp_report['ROC_AUC']:.3f}\n"
                 f"- PR AP: {mlp_report['PR_AP']:.3f}\n"
                 f"- Umbral F1*: {mlp_report['BestThresholdF1']:.2f}\n")

    if args.plots or args.calibration:
        lines.append("\n## Figuras\n")
        figs = []
        for name in ["LogisticRegression","DecisionTree","RandomForest","MLP"]:
            if args.plots:
                figs += [f"![ROC - {name}](figs/roc_{name}.png)",
                         f"![PR - {name}](figs/pr_{name}.png)",
                         f"![CM - {name}](figs/cm_{name}.png)"]
            if args.calibration:
                figs += [f"![Calibration - {name}](figs/cal_{name}.png)"]
        lines += [f"{ln}\n" for ln in figs]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Reporte: {report_path.resolve()}")
    print(f"[OK] Resultados guardados en: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
    