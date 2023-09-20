import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import torch
import torch.nn.functional as F
import torchvision.ops as vops

from slotformer.base_slots.models import to_rgb_from_tensor

FG_THRE = 0.5
PALETTE = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0),
           (255, 0, 255), (100, 100, 255), (200, 200, 100), (170, 120, 200),
           (255, 0, 0), (200, 100, 100), (10, 200, 100), (200, 200, 200),
           (50, 50, 50)]
PALETTE_np = np.array(PALETTE, dtype=np.uint8)
PALETTE_torch = torch.from_numpy(PALETTE_np).float() / 255. * 2. - 1.


def postproc_mask(batch_masks):
    """Post-process masks instead of directly taking argmax.

    Args:
        batch_masks: [B, T, N, 1, H, W]

    Returns:
        masks: [B, T, H, W]
    """
    batch_masks = batch_masks.clone()
    B, T, N, _, H, W = batch_masks.shape
    batch_masks = batch_masks.reshape(B * T, N, H * W)
    slots_max = batch_masks.max(-1)[0]  # [B*T, N]
    bg_idx = slots_max.argmin(-1)  # [B*T]
    spatial_max = batch_masks.max(1)[0]  # [B*T, H*W]
    bg_mask = (spatial_max < FG_THRE)  # [B*T, H*W]
    idx_mask = torch.zeros((B * T, N)).type_as(bg_mask)
    idx_mask[torch.arange(B * T), bg_idx] = True
    # set the background mask score to 1
    batch_masks[idx_mask.unsqueeze(-1) * bg_mask.unsqueeze(1)] = 1.
    masks = batch_masks.argmax(1)  # [B*T, H*W]
    return masks.reshape(B, T, H, W)


def masks_to_boxes_w_empty_mask(binary_masks):
    """binary_masks: [B, H, W]."""
    B = binary_masks.shape[0]
    obj_mask = (binary_masks.sum([-1, -2]) > 0)  # [B]
    bboxes = torch.ones((B, 4)).float().to(binary_masks.device) * -1.
    bboxes[obj_mask] = vops.masks_to_boxes(binary_masks[obj_mask])
    return bboxes


def masks_to_boxes(masks, num_boxes=7):
    """Convert seg_masks to bboxes.

    Args:
        masks: [B, T, H, W], output after taking argmax
        num_boxes: number of boxes to generate (num_slots)

    Returns:
        bboxes: [B, T, N, 4], 4: [x1, y1, x2, y2]
    """
    B, T, H, W = masks.shape
    binary_masks = F.one_hot(masks, num_classes=num_boxes)  # [B, T, H, W, N]
    binary_masks = binary_masks.permute(0, 1, 4, 2, 3)  # [B, T, N, H, W]
    binary_masks = binary_masks.contiguous().flatten(0, 2)
    bboxes = masks_to_boxes_w_empty_mask(binary_masks)  # [B*T*N, 4]
    bboxes = bboxes.reshape(B, T, num_boxes, 4)  # [B, T, N, 4]
    return bboxes


def mse_metric(x, y):
    """x/y: [B, C, H, W]"""
    # people often sum over spatial dimensions in video prediction MSE
    # see e.g. https://github.com/Yunbo426/predrnn-pp/blob/40764c52f433290aa02414b5f25c09c72b98e0af/train.py#L244
    return ((x - y)**2).sum(-1).sum(-1).mean()


def psnr_metric(x, y):
    """x/y: [B, C, H, W]"""
    psnrs = [
        peak_signal_noise_ratio(
            x[i],
            y[i],
            data_range=1.,
        ) for i in range(x.shape[0])
    ]
    return np.mean(psnrs)


def ssim_metric(x, y):
    """x/y: [B, C, H, W]"""
    x = x * 255.
    y = y * 255.
    ssims = [
        structural_similarity(
            x[i],
            y[i],
            channel_axis=0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=255,
        ) for i in range(x.shape[0])
    ]
    return np.mean(ssims)


def perceptual_dist(x, y, loss_fn):
    """x/y: [B, C, H, W]"""
    return loss_fn(x, y).mean()


def adjusted_rand_index(true_ids, pred_ids, ignore_background=False):
    """Computes the adjusted Rand index (ARI), a clustering similarity score.

    Code borrowed from https://github.com/google-research/slot-attention-video/blob/e8ab54620d0f1934b332ddc09f1dba7bc07ff601/savi/lib/metrics.py#L111

    Args:
        true_ids: An integer-valued array of shape
            [batch_size, seq_len, H, W]. The true cluster assignment encoded
            as integer ids.
        pred_ids: An integer-valued array of shape
            [batch_size, seq_len, H, W]. The predicted cluster assignment
            encoded as integer ids.
        ignore_background: Boolean, if True, then ignore all pixels where
            true_ids == 0 (default: False).

    Returns:
        ARI scores as a float32 array of shape [batch_size].
    """
    if len(true_ids.shape) == 3:
        true_ids = true_ids.unsqueeze(1)
    if len(pred_ids.shape) == 3:
        pred_ids = pred_ids.unsqueeze(1)

    true_oh = F.one_hot(true_ids).float()
    pred_oh = F.one_hot(pred_ids).float()

    if ignore_background:
        true_oh = true_oh[..., 1:]  # Remove the background row.

    N = torch.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
    A = torch.sum(N, dim=-1)  # row-sum  (batch_size, c)
    B = torch.sum(N, dim=-2)  # col-sum  (batch_size, k)
    num_points = torch.sum(A, dim=1)

    rindex = torch.sum(N * (N - 1), dim=[1, 2])
    aindex = torch.sum(A * (A - 1), dim=1)
    bindex = torch.sum(B * (B - 1), dim=1)
    expected_rindex = aindex * bindex / torch.clamp(
        num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator != 0, ari, torch.tensor(1.).type_as(ari))


def ARI_metric(x, y):
    """x/y: [B, H, W], both are seg_masks after argmax."""
    assert 'int' in str(x.dtype)
    assert 'int' in str(y.dtype)
    return adjusted_rand_index(x, y).mean().item()


def fARI_metric(x, y):
    """x/y: [B, H, W], both are seg_masks after argmax."""
    assert 'int' in str(x.dtype)
    assert 'int' in str(y.dtype)
    return adjusted_rand_index(x, y, ignore_background=True).mean().item()


def bbox_precision_recall(gt_pres_mask, gt_bbox, pred_bbox, ovthresh=0.5):
    """Compute the precision of predicted bounding boxes.

    Args:
        gt_pres_mask: A boolean tensor of shape [N]
        gt_bbox: A tensor of shape [N, 4]
        pred_bbox: A tensor of shape [M, 4]
    """
    gt_bbox, pred_bbox = gt_bbox.clone(), pred_bbox.clone()
    gt_bbox = gt_bbox[gt_pres_mask.bool()]
    pred_bbox = pred_bbox[pred_bbox[:, 0] >= 0.]
    N, M = gt_bbox.shape[0], pred_bbox.shape[0]
    assert gt_bbox.shape[1] == pred_bbox.shape[1] == 4
    # assert M >= N
    tp, fp = 0, 0
    bbox_used = [False] * pred_bbox.shape[0]
    bbox_ious = vops.box_iou(gt_bbox, pred_bbox)  # [N, M]

    # Find the best iou match for each ground truth bbox.
    for i in range(N):
        best_iou_idx = bbox_ious[i].argmax().item()
        best_iou = bbox_ious[i, best_iou_idx].item()
        if best_iou >= ovthresh and not bbox_used[best_iou_idx]:
            tp += 1
            bbox_used[best_iou_idx] = True
        else:
            fp += 1

    # compute precision and recall
    precision = tp / float(M)
    recall = tp / float(N)
    return precision, recall


def batch_bbox_precision_recall(gt_pres_mask, gt_bbox, pred_bbox):
    """Compute the precision of predicted bounding boxes over batch."""
    aps, ars = [], []
    for i in range(gt_pres_mask.shape[0]):
        ap, ar = bbox_precision_recall(gt_pres_mask[i], gt_bbox[i],
                                       pred_bbox[i])
        aps.append(ap)
        ars.append(ar)
    return np.mean(aps), np.mean(ars)


def hungarian_miou(gt_mask, pred_mask):
    """both mask: [H*W] after argmax, 0 is gt background index."""
    true_oh = F.one_hot(gt_mask).float()[..., 1:]  # only foreground, [HW, N]
    pred_oh = F.one_hot(pred_mask).float()  # [HW, M]
    N, M = true_oh.shape[-1], pred_oh.shape[-1]
    # compute all pairwise IoU
    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)  # [N, M]
    union = (true_oh.sum(0)[:, None] + pred_oh.sum(0)[None, :]) - intersect  # [N, M]
    iou = intersect / (union + 1e-8)  # [N, M]
    iou = iou.detach().cpu().numpy()
    # find the best match for each gt
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    # there are two possibilities here
    #   1. M >= N, just take the best match mean
    #   2. M < N, some objects are not detected, their iou is 0
    if M >= N:
        assert (row_ind == np.arange(N)).all()
        return iou[row_ind, col_ind].mean()
    return iou[row_ind, col_ind].sum() / float(N)


def miou_metric(gt_mask, pred_mask):
    """both mask: [B, H, W], both are seg_masks after argmax."""
    assert 'int' in str(gt_mask.dtype)
    assert 'int' in str(pred_mask.dtype)
    gt_mask, pred_mask = gt_mask.flatten(1, 2), pred_mask.flatten(1, 2)
    ious = [
        hungarian_miou(gt_mask[i], pred_mask[i])
        for i in range(gt_mask.shape[0])
    ]
    return np.mean(ious)


@torch.no_grad()
def pred_eval_step(
    gt,
    pred,
    lpips_fn,
    gt_mask=None,
    pred_mask=None,
    gt_pres_mask=None,
    gt_bbox=None,
    pred_bbox=None,
    eval_traj=True,
):
    """Both of shape [B, T, C, H, W], torch.Tensor.
    masks of shape [B, T, H, W].
    pres_mask of shape [B, T, N].
    bboxes of shape [B, T, N/M, 4].

    eval_traj: whether to evaluate the trajectory (measured by bbox and mask).

    Compute metrics for every timestep.
    """
    assert len(gt.shape) == len(pred.shape) == 5
    assert gt.shape == pred.shape
    assert gt.shape[2] == 3
    if eval_traj:
        assert len(gt_mask.shape) == len(pred_mask.shape) == 4
        assert gt_mask.shape == pred_mask.shape
    if eval_traj:
        assert len(gt_pres_mask.shape) == 3
        assert len(gt_bbox.shape) == len(pred_bbox.shape) == 4
    T = gt.shape[1]

    # compute perceptual dist & mask metrics before converting to numpy
    all_percept_dist, all_ari, all_fari, all_miou = [], [], [], []
    for t in range(T):
        one_gt, one_pred = gt[:, t], pred[:, t]
        percept_dist = perceptual_dist(one_gt, one_pred, lpips_fn).item()
        all_percept_dist.append(percept_dist)
        if eval_traj:
            one_gt_mask, one_pred_mask = gt_mask[:, t], pred_mask[:, t]
            ari = ARI_metric(one_gt_mask, one_pred_mask)
            fari = fARI_metric(one_gt_mask, one_pred_mask)
            miou = miou_metric(one_gt_mask, one_pred_mask)
            all_ari.append(ari)
            all_fari.append(fari)
            all_miou.append(miou)
        else:
            all_ari.append(0.)
            all_fari.append(0.)
            all_miou.append(0.)

    # compute bbox metrics
    all_ap, all_ar = [], []
    for t in range(T):
        if not eval_traj:
            all_ap.append(0.)
            all_ar.append(0.)
            continue
        one_gt_pres_mask, one_gt_bbox, one_pred_bbox = \
            gt_pres_mask[:, t], gt_bbox[:, t], pred_bbox[:, t]
        ap, ar = batch_bbox_precision_recall(one_gt_pres_mask, one_gt_bbox,
                                             one_pred_bbox)
        all_ap.append(ap)
        all_ar.append(ar)

    gt = to_rgb_from_tensor(gt).cpu().numpy()
    pred = to_rgb_from_tensor(pred).cpu().numpy()
    all_mse, all_ssim, all_psnr = [], [], []
    for t in range(T):
        one_gt, one_pred = gt[:, t], pred[:, t]
        mse = mse_metric(one_gt, one_pred)
        psnr = psnr_metric(one_gt, one_pred)
        ssim = ssim_metric(one_gt, one_pred)
        all_mse.append(mse)
        all_ssim.append(ssim)
        all_psnr.append(psnr)
    return {
        'mse': all_mse,
        'ssim': all_ssim,
        'psnr': all_psnr,
        'percept_dist': all_percept_dist,
        'ari': all_ari,
        'fari': all_fari,
        'miou': all_miou,
        'ap': all_ap,
        'ar': all_ar,
    }
