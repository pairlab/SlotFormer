from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 1  # lightweight since we don't use img recon loss
    max_epochs = 50  # ~300k steps
    save_interval = 0.2  # save every 0.2 epoch
    eval_interval = 5  # evaluate every 5 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 25  # because we have 25 tasks in PHYRE

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 2e-4
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay, no gradient clipping

    # data settings
    dataset = 'phyre_slots'
    data_root = './data/PHYRE'
    slots_root = 'checkpoint/savi_phyre_params-fold0/{}_slots'  # in a folder
    frame_offset = 1  # useless, just for compatibility
    fps = 1  # simulate 1 FPS
    n_sample_frames = (1 + 10) * fps  # 1 burn-in, 10 rollout
    video_len = 11 * fps

    # PHYRE-related configs, see `savi_phyre_params-fold0.py` for details
    phyre_protocal = 'within'
    phyre_fold = 0
    data_ratio = 0.1
    pos_ratio = 0.2
    reverse_color = True

    train_batch_size = 64 // gpus
    val_batch_size = 8  # since we recon img in eval
    num_workers = 8

    # model configs
    model = 'SingleStepSlotFormer'
    resolution = (128, 128)
    input_frames = 1  # PHYRE rollouts only condition on the first frame

    num_slots = 8
    slot_size = 128
    slot_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
    )

    # Rollouter
    rollout_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        history_len=input_frames,
        cond_len=6,  # this is the real `history_len`
        t_pe='sin',  # sine temporal P.E.
        slots_pe='',  # no slots P.E.
        # Transformer-related configs
        d_model=slot_size * 2,
        num_layers=8,
        num_heads=8,
        ffn_dim=slot_size * 2 * 4,
        norm_first=True,
    )

    # CNN Decoder
    dec_dict = dict(
        dec_channels=(128, 64, 64, 64, 64),
        dec_resolution=(16, 16),
        dec_ks=5,
        dec_norm='',
        dec_ckp_path='pretrained/savi_phyre_params-fold0/model_30.pth',
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=False,  # dec_res 16 makes decoding memory-intensive
        # and not very helpful for learning dynamics
    )

    # temporally weighting the loss as done in RPIN?
    # we don't extensively ablate in our experiments
    # in some fold it helps, in some fold it's slightly worse
    use_loss_decay = False
    loss_decay_pct = 0.8  # decay in the first 80% of total training steps

    slot_recon_loss_w = 1.
    img_recon_loss_w = 0.1
