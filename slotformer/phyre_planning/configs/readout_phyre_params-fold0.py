from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    # the training of this linear readout model is very fast
    gpus = 1
    max_epochs = 50
    eval_interval = 2
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 25  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-3
    warmup_steps_pct = 0.1

    # data settings
    dataset = 'phyre_rollout_slots'
    data_root = './data/PHYRE'
    slot_root = 'checkpoint/slotformer_phyre_params-fold0/{}_slots'
    frame_offset = 1
    fps = 1
    n_sample_frames = 11 * fps
    video_len = 11 * fps

    # PHYRE-related configs, see `savi_phyre_params-fold0.py` for details
    phyre_protocal = 'within'
    phyre_fold = 0
    data_ratio = 0.1
    pos_ratio = 0.2
    reverse_color = True

    train_batch_size = 256 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'PHYREReadout'
    resolution = (128, 128)

    # SAVi on PHYRE uses 8 slots, each with 128-dim
    slot_size = 128
    readout_dict = dict(
        num_slots=8,
        slot_size=slot_size,
        t_pe='sin',
        d_model=slot_size,
        num_layers=4,
        num_heads=8,
        ffn_dim=slot_size * 4,
        norm_first=True,
        sel_slots=[0, 3],
    )

    vqa_loss_w = 1.
