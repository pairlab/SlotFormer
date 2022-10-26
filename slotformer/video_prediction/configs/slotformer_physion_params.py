from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 1  # lightweight since we don't use img recon loss
    max_epochs = 25  # ~230k steps
    save_interval = 0.125  # save every 0.125 epoch
    eval_interval = 2  # evaluate every 2 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 8  # Physion has 8 scenarios

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 2e-4
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay, no gradient clipping

    # data settings
    dataset = 'physion_slots_training'
    data_root = './data/Physion'
    slots_root = './data/Physion/training_slots.pkl'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 15 + 10  # 15 burn-in, 10 rollout
    frame_offset = 3  # subsample every 3 frames to increase difference
    video_len = 150  # take the first 150 frames of each video
    train_batch_size = 128 // gpus
    val_batch_size = train_batch_size
    num_workers = 8

    # model configs
    model = 'STEVESlotFormer'
    resolution = (128, 128)
    input_frames = 15  # burn-in frames

    num_slots = 6
    slot_size = 192
    slot_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
    )

    # Rollouter
    rollout_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        history_len=input_frames,
        t_pe='sin',  # sine temporal P.E.
        slots_pe='',  # no slots P.E.
        # Transformer-related configs
        d_model=256,
        num_layers=8,
        num_heads=8,
        ffn_dim=256 * 4,
        norm_first=True,
    )

    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=4096,
        dvae_ckp_path='pretrained/dvae_physion_params/model_20.pth',
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
        dec_ckp_path='pretrained/steve_physion_params/model_10.pth',
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=False,  # STEVE recon img is too memory-intensive
    )

    slot_recon_loss_w = 1.
    img_recon_loss_w = 1.
