import numpy as np
import torch
import torch.nn as nn
from text_detect_torch.east import cfg

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def quad_norm(g_true):
    g_shape = g_true.size()
    delta_xy_matrix = torch.reshape(g_true, [-1, 2, 2])
    diff = delta_xy_matrix[:, 0:1, :] - delta_xy_matrix[:, 1:2, :]
    square = diff ** 2
    distance = torch.sqrt(torch.sum(square, dim=-1))
    distance = distance * 4.0
    distance = distance + cfg.epsilon
    distance = torch.reshape(distance, g_shape[:-1])
    return distance


def smooth_l1_loss(preddiction_tensor, target_tensor, weights):
    n_q = torch.reshape(quad_norm(target_tensor), weights.size())
    pixel_wise_smooth_l1norm = torch.nn.SmoothL1Loss(reduction='none')(preddiction_tensor, target_tensor)
    pixel_wise_smooth_l1norm = (torch.sum(pixel_wise_smooth_l1norm, dim=-1) / n_q) * weights
    return pixel_wise_smooth_l1norm


def quad_loss(y_true, y_pred):
    if y_true.size(1) == 7:
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)
    # inside_score_loss
    logits = y_pred[:, :, :, :1]
    labels = y_true[:, :, :, :1]
    beta = 1 - torch.mean(labels)
    predicts = nn.Sigmoid().to(device)(logits)
    inside_score_loss = torch.mean(
        -1 * (beta * labels * torch.log(predicts + cfg.epsilon) +
              (1 - beta) * (1 - labels) * torch.log(1 - predicts + cfg.epsilon)))
    inside_score_loss = inside_score_loss * cfg.lambda_inside_score_loss
    # side_v_code_loss
    v_logits = y_pred[:, :, :, 1:3]
    v_labels = y_true[:, :, :, 1:3]
    v_beta = 1 - (torch.mean(y_true[:, :, :, 1:2]) /
                  (torch.mean(labels) + cfg.epsilon))
    v_predicts = nn.Sigmoid().to(device)(v_logits)
    pos = -1 * v_beta * v_labels * torch.log(v_predicts + cfg.epsilon)
    neg = -1 * (1 - v_beta) * (1 - v_labels) * torch.log(1 - v_predicts + cfg.epsilon)
    positive_weights = torch.eq(y_true[:, :, :, 0], 1).float()
    side_v_code_loss = torch.sum(torch.sum(pos + neg, dim=-1) * positive_weights) / (
            torch.sum(positive_weights) + cfg.epsilon)
    side_v_code_loss = side_v_code_loss * cfg.lambda_side_v_code_loss
    # side_v_coord_loss
    g_logits = y_pred[:, :, :, 3:]
    g_labels = y_true[:, :, :, 3:]
    v_weights = torch.eq(y_true[:, :, :, 1], 1).float()
    pixel_wise_smooth_l1norm = smooth_l1_loss(g_logits, g_labels, v_weights)
    side_v_coord_loss = torch.sum(pixel_wise_smooth_l1norm) / (
        torch.sum(v_weights) + cfg.epsilon)
    side_v_coord_loss = side_v_coord_loss * cfg.lambda_side_v_coord_loss
    return inside_score_loss + side_v_code_loss + side_v_coord_loss


