import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
from typing import List, Tuple


# === 1) Funzione per costruire la curva “Cost Search” sul test set ===

def compute_cost_search_curve(y_true: np.ndarray, y_logits: np.ndarray, num_steps: int = 100):
    """
    Genera due liste (fn_perc_list, neg_pred_list) scorrendo 'num_steps' soglie uniformi tra 0 e 1.
    - y_true: array di etichette vere (0/1)
    - y_logits: array di probabilità (score) per la classe 1
    Restituisce:
      thresholds: lista di soglie usate
      fn_perc_list: false_negative % per ogni soglia
      neg_pred_list: negative_predicted % per ogni soglia
    """
    thresholds = np.linspace(0.0, 1.0, num_steps)
    fn_perc_list = []
    neg_pred_list = []

    total = len(y_true)
    for t in thresholds:
        # 1) Binarizza con threshold t
        y_pred = (y_logits >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # 2) Calcola FN% e NP%
        fn_perc = fn / total
        neg_pred_perc = (tn + fn) / total

        fn_perc_list.append(fn_perc)
        neg_pred_list.append(neg_pred_perc)

    return thresholds, fn_perc_list, neg_pred_list


# === 2) Funzione di plotting per il test set ===

def plot_cost_search_test(
    thresholds: List[float],
    fn_perc_list: List[float],
    neg_pred_list: List[float],
    utopian: Tuple[float, float],
    graph_title: str,
    x_axis_name: str,
    y_axis_name: str
) -> go.Figure:
    """
    Disegna:
      - curva blu “Cost Search” (linea+marker) per ciascun threshold
      - punto rosso utopico in (0, utopian[1])
    """
    fig = go.Figure()

    # 1) Curva “Cost Search”
    # Non stampiamo i thresholds come hover‐text per ogni marker (troppi punti), ma solo ogni ~5° punto:
    hover_text = [
        f"Threshold: {t:.2f}<br>FN%: {fn_perc_list[i]:.3f}<br>NP%: {neg_pred_list[i]:.3f}"
        for i, t in enumerate(thresholds)
    ]
    fig.add_trace(go.Scatter(
        x=fn_perc_list,
        y=neg_pred_list,
        mode='lines+markers',
        marker=dict(color='#636EFA', size=5),
        line=dict(color='#636EFA', width=2),
        name='Cost Search (test)',
        text=hover_text,
        hovertemplate="%{text}<extra></extra>"
    ))

    # 2) Punto utopico (rosso) a x=0, y=utopian[1]
    fig.add_trace(go.Scatter(
        x=[utopian[0]],
        y=[utopian[1]],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Utopian point',
        hovertemplate=f"FN%: 0.000<br>NP%: {utopian[1]:.3f}<extra></extra>"
    ))

    # 3) Layout con sfondo chiaro
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


# === 3) Parte principale dello script ===

if __name__ == "__main__":
    # 3.1) PERCORSI: 
    #     Inserisci qui il percorso al tuo file di test predictions.parquet
    test_pred_path = "/Users/ludole/Desktop/all results/os_months_24/classification/cross_validation/hypothesis_driven/pyrad-noimp/rwd/seed_0/fold_GHD/eval/00000-mb_attention_mil/predictions.parquet"
    #     Se hai anche score_test.csv relativo al test, puoi leggerlo, ma per la curva non serve.

    # 3.2) Carica il test set
    if not os.path.isfile(test_pred_path):
        raise FileNotFoundError(f"Non trovo {test_pred_path} - controlla il percorso!")

    df_test = pd.read_parquet(test_pred_path)
    # df_test deve contenere almeno:
    #   - "y_true" (0/1)
    #   - "y_pred1" (probabilità per la classe 1)
    y_true_test = df_test["y_true"].to_numpy()
    y_logits_test = df_test["y_pred1"].to_numpy()

    # 3.3) Calcola la curva “Cost Search” scorrendo 100 soglie da 0 a 1
    thresholds, fn_perc_list, neg_pred_list = compute_cost_search_curve(
        y_true=y_true_test,
        y_logits=y_logits_test,
        num_steps=100
    )

    # 3.4) Costruisci il punto utopico: x=0, y=frazione di negativi nel test set
    total_test = len(y_true_test)
    utopian_y = np.sum(y_true_test == 0) / total_test
    utopian_point = (0.0, utopian_y)

    # 3.5) Plotta tutto
    fig = plot_cost_search_test(
        thresholds     = thresholds,
        fn_perc_list   = fn_perc_list,
        neg_pred_list  = neg_pred_list,
        utopian        = utopian_point,
        graph_title    = "Cost Search (Test Set)",
        x_axis_name    = "False Negative Percentage",
        y_axis_name    = "Negative Predicted Percentage"
    )
    fig.show()
