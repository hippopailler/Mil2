import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go

# === 1) Funzione per calcolare la curva “Cost Search” per un singolo array y_true/y_logits ===

def compute_cost_search_curve(y_true, y_logits, num_steps=100):
    """
    Data una fold:
      - y_true: array di 0/1
      - y_logits: array di score (probabilità classe 1)
    Restituisce:
      thresholds       : lista di soglie da 0.0 a 1.0
      fn_perc_list     : lista di FN% corrispondenti
      neg_pred_list    : lista di NegativePred% corrispondenti
    """
    thresholds = np.linspace(0.0, 1.0, num_steps)
    fn_perc_list = []
    neg_pred_list = []
    total = len(y_true)

    for t in thresholds:
        y_pred = (y_logits >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fn_perc = fn / total
        neg_pred = (tn + fn) / total
        fn_perc_list.append(fn_perc)
        neg_pred_list.append(neg_pred)

    return thresholds, fn_perc_list, neg_pred_list


# === 2) Funzione di plotting per tutte le curve dei fold ===

def plot_all_fold_curves(
    curves_dict,
    utopian_points,
    graph_title="Cost Search di tutti i fold",
    x_axis_name="False Negative %",
    y_axis_name="Negative Predicted %"
):
    """
    curves_dict: dizionario con chiave = "seed_x/fold_y" e valore = (fn_perc_list, neg_pred_list, thresholds)
    utopian_points: dizionario con chiave = "seed_x/fold_y" e valore = utopian_y (float)
    Disegna una linea+marker per ogni fold in curves_dict, e un punto rosso per il corrispondente utopian.
    """
    fig = go.Figure()

    # 1) Aggiungo tutte le curve blu (una per fold)
    for fold_id, (fn_perc_list, neg_pred_list, thresholds) in curves_dict.items():
        # Preparo un hover‐text che mostri la soglia solo su alcuni punti (per non sovraccaricare)
        hover_text = [
            f"Fold: {fold_id}<br>Threshold: {thresholds[i]:.2f}<br>"
            f"FN%: {fn_perc_list[i]:.3f}<br>NP%: {neg_pred_list[i]:.3f}"
            for i in range(len(thresholds))
        ]

        fig.add_trace(go.Scatter(
            x=fn_perc_list,
            y=neg_pred_list,
            mode='lines',
            line=dict(width=2),
            name=fold_id,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>"
        ))

    # 2) Aggiungo, per ciascun fold, il punto utopico (rosso) in x=0, y=utopian_y
    #    Lo disegno solo una volta per fold (con marker size piccolo).
    for fold_id, ut_y in utopian_points.items():
        fig.add_trace(go.Scatter(
            x=[0.0],
            y=[ut_y],
            mode='markers',
            marker=dict(color='red', size=8),
            name=f"Utopia {fold_id}",
            hovertemplate=f"Fold: {fold_id}<br>FN%: 0.000<br>NP%: {ut_y:.3f}<extra></extra>"
        ))

    # 3) Layout con sfondo chiaro
    fig.update_layout(
        title=graph_title,
        xaxis_title=x_axis_name,
        yaxis_title=y_axis_name,
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(
            x=1.01, y=1.0,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1
        )
    )

    return fig


# === 3) Script principale: cerca tutte le fold nei seed e costruisci le curve ===

if __name__ == "__main__":
    # 3.1) Modifica qui: cartella che contiene tutti i seed (es. "rwd")
    root_model_path = "/Users/ludole/Desktop/all results/os_months_24/classification/cross_validation/hypothesis_driven/pyrad-noimp/rwd"

    # Controllo che la cartella esista
    if not os.path.isdir(root_model_path):
        raise FileNotFoundError(f"Non trovo la cartella dei seed: {root_model_path}")

    curves_dict = {}       # { "seed_i/fold_j": (fn_perc_list, neg_pred_list, thresholds) }
    utopian_points = {}    # { "seed_i/fold_j": utopian_y }

    # 3.2) Ciclo su ciascun seed_* dentro root_model_path
    for seed_name in sorted(os.listdir(root_model_path)):
        seed_path = os.path.join(root_model_path, seed_name)
        if not os.path.isdir(seed_path):
            continue

        # 3.3) Dentro ogni seed, cerco le cartelle fold_*
        for fold_name in sorted(os.listdir(seed_path)):
            fold_path = os.path.join(seed_path, fold_name)
            eval_path = os.path.join(fold_path, "eval/00000-mb_attention_mil")
            if not os.path.isdir(eval_path):
                continue

            fold_id = f"{seed_name}/{fold_name}"

            # 3.4) Leggo predictions.parquet
            parquet_file = os.path.join(eval_path, "predictions.parquet")
            if not os.path.isfile(parquet_file):
                print(f"⚠️ Warning: manca {parquet_file}, salto {fold_id}")
                continue

            df_fold = pd.read_parquet(parquet_file)
            y_true = df_fold["y_true"].to_numpy()
            y_logits = df_fold["y_pred1"].to_numpy()

            # 3.5) Calcolo la curva Cost Search per questa fold
            thresholds, fn_perc_list, neg_pred_list = compute_cost_search_curve(y_true, y_logits, num_steps=100)
            curves_dict[fold_id] = (fn_perc_list, neg_pred_list, thresholds)

            # 3.6) Calcolo utopian_y per questa fold (frazione di negativi nella fold)
            total = len(y_true)
            ut_y = np.sum(y_true == 0) / total
            utopian_points[fold_id] = ut_y

    # 3.7) Controllo di aver trovato almeno una curva
    if not curves_dict:
        raise RuntimeError("Non ho trovato alcuna fold valida: controlla la struttura delle cartelle.")

    # 3.8) Plotto tutte le curve + relativi utopian point
    fig = plot_all_fold_curves(
        curves_dict    = curves_dict,
        utopian_points = utopian_points,
        graph_title    = "Cost Search: tutte le curve di tutti i fold",
        x_axis_name    = "False Negative %",
        y_axis_name    = "Negative Predicted %"
    )
    fig.show()
