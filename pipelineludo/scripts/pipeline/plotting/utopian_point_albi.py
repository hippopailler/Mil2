import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from typing import List, Dict, Optional

# === 1) Definizione semplificata delle classi Point e Line ===

class Point:
    """
    Rappresenta un singolo punto (es. il punto utopico).
    """
    def __init__(self, x, y, color='red'):
        self.x = x
        self.y = y
        self.color = color


class Line:
    """
    Rappresenta la curva CV di un solo modello:
      - x_values, y_values: liste di float
      - color: stringa HEX
      - attributes: dizionario (opzionale) di liste di float (es. F1, AUC, ecc.)
    """
    def __init__(
        self,
        x_values: List[float],
        y_values: List[float],
        color: str = '#636EFA',
        attributes: Optional[Dict[str, List[float]]] = None
    ):
        self.x_values = x_values
        self.y_values = y_values
        self.color = color
        self.attributes = attributes or {}


# === 2) Funzione di plotting per un singolo modello ===

def plot_cost_search_single(
    utopian: Point,
    cv_curve: Line,
    graph_title: str,
    x_axis_name: str,
    y_axis_name: str
) -> go.Figure:
    """
    Disegna:
      - un punto rosso (utopico) in (utopian.x, utopian.y)
      - una curva blu+marker per la CV di un solo modello, con hover‐text preso da cv_curve.attributes
    """
    fig = go.Figure()

    # 2.1) Punto utopico (rosso)
    fig.add_trace(go.Scatter(
        x=[utopian.x],
        y=[utopian.y],
        mode='markers',
        marker=dict(color=utopian.color, size=12),
        name='Utopian point'
    ))

    # 2.2) Curva CV (blu) con hover‐text
    # Costruiamo la lista di stringhe per ogni fold:
    hover_text = [
        "<br>".join(f"{k}: {v:.3f}" for k, v in zip(cv_curve.attributes.keys(), combo))
        for combo in zip(*cv_curve.attributes.values())
    ]
    fig.add_trace(go.Scatter(
        x=cv_curve.x_values,
        y=cv_curve.y_values,
        mode='lines+markers',
        marker=dict(color=cv_curve.color, size=6),
        line=dict(color=cv_curve.color, width=2),
        name='RWD CV',
        text=hover_text,
        hovertemplate="%{text}<extra></extra>"
    ))

    # 2.3) Layout con sfondo chiaro
    fig.update_layout(
        title=graph_title,
        xaxis_title=x_axis_name,
        yaxis_title=y_axis_name,
        template='plotly_white',
        width=800,
        height=500,
        legend=dict(x=0.75, y=0.95, bgcolor="rgba(255,255,255,0.8)")
    )
    return fig


# === 3) Parte principale: raccogliere i dati e chiamare il plotting ===

if __name__ == "__main__":
    # 3.1) Cambia questo percorso con la cartella rwd/seed_0 corretta
    base_path = "/Users/ludole/Desktop/all results/os_months_24/classification/cross_validation/hypothesis_driven/pyrad-noimp/rwd/seed_0"

    # Liste per raccogliere i risultati per ogni fold
    y_true_all = []
    fn_perc_list = []
    neg_pred_list = []
    f1_list = []
    auc_list = []
    sens_list = []
    spec_list = []

    # 3.2) Itera su ogni cartella “fold_*” all’interno di rwd/seed_0
    for fold_name in sorted(os.listdir(base_path)):
        fold_path = os.path.join(base_path, fold_name)
        eval_path = os.path.join(fold_path, "eval/00000-mb_attention_mil")
        if not os.path.isdir(eval_path):
            continue

        # Legge predictions.parquet
        preds_df = pd.read_parquet(os.path.join(eval_path, "predictions.parquet"))
        y_true = preds_df["y_true"].to_numpy()
        y_logits = preds_df["y_pred1"].to_numpy()
        y_true_all.append(y_true)

        # Legge score_test.csv (colonne: F1, AUC, Sensitivity, Specificity, N)
        scores_df = pd.read_csv(os.path.join(eval_path, "scores_test.csv"))
        f1_list.append(scores_df["F1"].iloc[0])
        auc_list.append(scores_df["AUC"].iloc[0])
        sens_list.append(scores_df["Sensitivity"].iloc[0])
        spec_list.append(scores_df["Specificity"].iloc[0])

        # Trasforma logits → predizioni binarie (soglia 0.5)
        y_pred_bin = (y_logits >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
        total = len(y_true)

        # False‐negative % e Negative‐predicted %
        fn_perc_list.append(fn / total)
        neg_pred_list.append((tn + fn) / total)

    # 3.3) Costruisci il punto utopico condiviso: x=0, y=frazione di negativi in tutte le y_true
    all_y = np.concatenate(y_true_all)
    utopian_y = np.sum(all_y == 0) / len(all_y)
    utopian_pt = Point(x=0.0, y=utopian_y, color='red')

    # 3.4) Costruisci la “curva CV” con le metriche per hover‐text
    metrics = {
        "F1":           f1_list,
        "AUC":          auc_list,
        "Sensitivity":  sens_list,
        "Specificity":  spec_list
    }
    rwd_curve = Line(
        x_values=fn_perc_list,
        y_values=neg_pred_list,
        color='#636EFA',
        attributes=metrics
    )

    # 3.5) Genera e mostra la figura
    fig = plot_cost_search_single(
        utopian=utopian_pt,
        cv_curve=rwd_curve,
        graph_title="Cost Search (RWD OS_Months_24 CV)",
        x_axis_name="False Negative %",
        y_axis_name="Negative Predicted %"
    )
    fig.show()
