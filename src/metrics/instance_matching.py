"""
Métricas de detecção de instâncias (núcleos) via matching por IoU.

Usadas tanto na Parte 1 (instâncias vindas de componentes conexos)
quanto na Parte 2 (instâncias vindas do decode watershed) -- o
protocolo de avaliação é o mesmo, só muda como pred_instances é gerado.
"""

import numpy as np


def _instance_ids(mask):
    """Retorna os IDs de instância presentes numa máscara (H, W), excluindo o fundo (0)."""
    ids = np.unique(mask)
    return ids[ids != 0]


def compute_iou_matrix(pred_mask, gt_mask):
    """
    Calcula a matriz de IoU entre todas as instâncias preditas e todas as
    instâncias reais.

    pred_mask, gt_mask: arrays (H, W) int, 0 = fundo, 1..N = ID de instância.
    Retorna: matriz (n_pred, n_gt) com o IoU de cada par (pred_i, gt_j).
    """
    pred_ids = _instance_ids(pred_mask)
    gt_ids = _instance_ids(gt_mask)

    n_pred, n_gt = len(pred_ids), len(gt_ids)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)

    for i, p_id in enumerate(pred_ids):
        pred_obj = pred_mask == p_id
        for j, g_id in enumerate(gt_ids):
            gt_obj = gt_mask == g_id
            intersection = np.logical_and(pred_obj, gt_obj).sum()
            if intersection == 0:
                continue  # já fica 0.0, evita trabalho à toa
            union = np.logical_or(pred_obj, gt_obj).sum()
            iou_matrix[i, j] = intersection / union

    return iou_matrix, pred_ids, gt_ids


def match_instances(iou_matrix, threshold):
    """
    Faz o matching 1-para-1 guloso entre predições e GTs para um dado
    limiar de IoU.

    Estratégia: ordena todos os pares (pred, gt) por IoU decrescente e vai
    aceitando o par se IoU >= threshold e nem pred nem gt já foram usados
    -- é a abordagem padrão (greedy matching), simples e determinística.

    Retorna: (n_tp, n_fp, n_fn)
        n_tp = pares casados com IoU >= threshold
        n_fp = predições sem par (incluindo "não existe nenhum GT")
        n_fn = GTs sem par
    """
    n_pred, n_gt = iou_matrix.shape

    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    # lista de (iou, pred_idx, gt_idx) ordenada do maior IoU para o menor
    pairs = [
        (iou_matrix[i, j], i, j)
        for i in range(n_pred)
        for j in range(n_gt)
        if iou_matrix[i, j] >= threshold
    ]
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred, matched_gt = set(), set()
    n_tp = 0
    for iou, i, j in pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        n_tp += 1

    n_fp = n_pred - len(matched_pred)
    n_fn = n_gt - len(matched_gt)
    return n_tp, n_fp, n_fn


def average_precision_at_threshold(pred_mask, gt_mask, threshold):
    """
    Precisão de detecção (não é a AP de ranking do COCO clássico, é a
    métrica de precisão usada na competição DSB2018):
        AP_t = TP / (TP + FP + FN)
    para um único limiar de IoU t.
    """
    iou_matrix, _, _ = compute_iou_matrix(pred_mask, gt_mask)
    n_tp, n_fp, n_fn = match_instances(iou_matrix, threshold)
    denom = n_tp + n_fp + n_fn
    if denom == 0:
        return 1.0  # nenhuma instância predita nem real: caso trivial
    return n_tp / denom


def mean_average_precision(pred_mask, gt_mask, thresholds=None):
    """
    mAP no protocolo 0.50:0.05:0.95 (padrão COCO / DSB2018): calcula a
    precisão de detecção em cada limiar de IoU e tira a média.
    """
    if thresholds is None:
        thresholds = np.arange(0.50, 1.00, 0.05)  # 0.50, 0.55, ..., 0.95

    precisions = [
        average_precision_at_threshold(pred_mask, gt_mask, t) for t in thresholds
    ]
    return float(np.mean(precisions))


def count_error(pred_mask, gt_mask):
    """Erro absoluto de contagem: |n_pred - n_gt|."""
    n_pred = len(_instance_ids(pred_mask))
    n_gt = len(_instance_ids(gt_mask))
    return abs(n_pred - n_gt)