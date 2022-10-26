from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    # the training of this linear readout model is very fast
    gpus = 1
    max_epochs = 50
    eval_interval = 5
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 8  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay without Warmup
    optimizer = 'Adam'
    lr = 1e-3
    warmup_steps_pct = 0.  # no warmup

    # data settings
    dataset = 'physion_slots_label_readout'  # fit on readout set
    data_root = './data/Physion'
    slots_root = 'checkpoint/slotformer_physion_params/readout_slots.pkl'
    tasks = ['all']
    n_sample_frames = 6  # useless
    frame_offset = 1  # take all video frames
    # we only take the first 75 frames of each video
    # due to error accumulation in the rollout, models trained on all frames
    # will overfit to some artifacts in later frames
    video_len = 75
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'PhysionReadout'
    resolution = (128, 128)

    # STEVE on Physion uses 6 slots, each with 192-dim
    slot_size = 192
    readout_dict = dict(
        num_slots=6,
        slot_size=slot_size,
        agg_func='max',
        feats_dim=slot_size,
    )

    vqa_loss_w = 1.
