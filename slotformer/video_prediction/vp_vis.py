import torch
import torchvision.utils as vutils

from vp_utils import to_rgb_from_tensor, PALETTE


def add_boundary(img, width=2, color='red'):
    """Add boundary to indicate GT or rollout frames.

    Args:
        img: (T, 3, H, W)
        width:
        color:

    Returns:
        img: (T, 3, H, W)
    """
    assert color in ['red', 'green']
    T, C, H, W = img.shape
    empty = torch.zeros((T, C, H + width * 2, W + width * 2)).type_as(img)
    if color == 'red':
        empty[:, 0] = 0.7
    else:
        empty[:, 1] = 0.7
    empty[:, :, width:-width, width:-width] = img
    return empty


def make_video(video, pred_video, history_len=6):
    """videos are of shape [T, C, H, W]"""
    T = video.shape[0]
    video = to_rgb_from_tensor(video)
    pred_video = to_rgb_from_tensor(pred_video)

    # add boundary
    video = add_boundary(video, color='green')
    pred_video = torch.cat([
        add_boundary(pred_video[:history_len], color='green'),
        add_boundary(pred_video[history_len:], color='red'),
    ],
                           dim=0)
    out = torch.stack([video, pred_video], dim=1)  # [T, 2, 3, H, W]
    save_video = torch.stack([
        vutils.make_grid(
            out[i].cpu(),
            nrow=1,
            padding=0,
        ) for i in range(T)
    ])  # [T, 3, 2*H, W]
    return save_video


def draw_bbox(img, bbox, bbox_width=2):
    """Draw bbox on images.

    Args:
        img: (3, H, W), torch.Tensor
        bbox: (N, 4)
    """
    N = bbox.shape[0]
    img = torch.round((to_rgb_from_tensor(img) * 255.)).to(dtype=torch.uint8)
    bbox = bbox.clone()
    bbox = bbox[bbox[:, 0] >= 0.]
    bbox_img = vutils.draw_bounding_boxes(
        img, bbox, colors=PALETTE[:N], width=bbox_width)
    bbox_img = bbox_img.float() / 255. * 2. - 1.
    return bbox_img


def batch_draw_bbox(imgs, bboxes, pres_masks=None, bbox_width=2):
    """Draw a batch of bbox on a batch images.

    We only draw bbox that is not padded indicated by `pres_mask`.

    Args:
        imgs: (B, 3, H, W), torch.Tensor
        bboxes: (B, N, 4), torch.Tensor
        pres_masks: (B, N), torch.Tensor

    Returns:
        imgs: (B, 3, H, W), torch.Tensor in uint8 can be directly saved.
    """
    imgs, bboxes = imgs.cpu(), bboxes.cpu()
    B, N = bboxes.shape[:2]
    if pres_masks is None:
        pres_masks = torch.ones((B, N)).bool().to(bboxes.device)
    else:
        pres_masks = pres_masks.cpu()
    bbox_imgs = []
    for i in range(B):
        img = imgs[i]
        bbox = bboxes[i]
        pres_mask = pres_masks[i].bool()
        img = draw_bbox(img, bbox[pres_mask], bbox_width=bbox_width)
        bbox_imgs.append(img)
    return torch.stack(bbox_imgs, dim=0)
