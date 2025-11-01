import torch as th
from sklearn.metrics import (auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve, roc_curve, roc_auc_score)

import numpy as np
import torch.nn.functional as fn
import numpy as np
import torch as th
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate2(args, model, graph_data,
              drug_graph, drug_feat, drug_sim_feat,
              dis_graph, dis_feat, dis_sim_feat, subgraphs):
    rating_values = graph_data['test'][2]
    enc_graph = graph_data['test'][0].int().to(args.device)
    dec_graph = graph_data['test'][1].int().to(args.device)

    model.eval()
    with th.no_grad():
        pred_ratings, _, _, _, _ = model(
            enc_graph, dec_graph,
            drug_graph, drug_sim_feat, drug_feat,
            dis_graph, dis_sim_feat, dis_feat,
            subgraphs
        )

    y_score = pred_ratings.view(-1).cpu().tolist()
    y_true = rating_values.cpu().tolist()

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)

    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall_curve, precision_curve)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    y_prob = sigmoid(np.array(y_score))
    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_true, y_pred)
    precision_cls = precision_score(y_true, y_pred)
    recall_cls = recall_score(y_true, y_pred)

    return auc, aupr, f1, precision_cls, recall_cls, y_true, y_score