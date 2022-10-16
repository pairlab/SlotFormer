from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 4  # 2 GPUs should also be good
    max_epochs = 12  # 230k steps
    save_interval = 0.2  # save every 0.2 epoch
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 5  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4  # a small learning rate is very important for SAVi training
    clip_grad = 0.05  # following the paper
    warmup_steps_pct = 0.025  # warmup in the first 2.5% of total steps

    # data settings
    dataset = 'clevrer'
    data_root = './data/CLEVRER'
    n_sample_frames = 6  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    filter_enter = False  # no need to filter videos when training SAVi
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'StoSAVi'  # stochastic version of SAVi
    resolution = (64, 64)
    input_frames = n_sample_frames

    # Slot Attention
    slot_dict = dict(
        num_slots=7,  # at most 6 objects per scene
        slot_size=128,
        slot_mlp_size=256,
        num_iterations=2,
        kernel_mlp=False,
    )

    # CNN Encoder
    enc_dict = dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=128,
        enc_norm='',
    )

    # CNN Decoder
    dec_dict = dict(
        dec_channels=(128, 64, 64, 64, 64),
        dec_resolution=(8, 8),
        dec_ks=5,
        dec_norm='',
    )

    # Predictor
    pred_dict = dict(
        pred_type='mlp',  # less information fusion to avoid slots sharing objs
        pred_rnn=False,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=slot_dict['slot_size'] * 4,
        pred_sg_every=None,
    )

    # loss configs
    loss_dict = dict(
        use_post_recon_loss=True,
        kld_method='var-0.01',  # prior Gaussian variance is 0.01
    )

    post_recon_loss_w = 1.  # posterior slots image recon
    kld_loss_w = 1e-4  # kld on kernels distribution
