from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 2
    max_epochs = 30  # 370k iters
    save_interval = 0.2  # save every 0.2 epoch
    eval_interval = 2  # evaluate every 2 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 25  # because we have 25 tasks in PHYRE

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4  # a small learning rate is very important for SAVi training
    clip_grad = 0.05  # following the paper
    warmup_steps_pct = 0.025  # warmup in the first 2.5% of total steps

    # data settings
    dataset = 'phyre'
    data_root = './data/PHYRE'
    n_sample_frames = 6  # train on video clips of 6 frames
    fps = 1  # simulate 1 FPS
    video_len = 15 * fps  # fix the maximum length of a simulation
    frame_offset = 1  # useless, just for compatibility
    # PHYRE related settings
    # PHYRE has 2 protocols: 'within' and 'cross'
    # each protocol should follow a 10-fold cross validation
    phyre_protocal = 'within'
    phyre_fold = 0
    data_ratio = 0.1  # we only use 10% of the data for training
    pos_ratio = 0.2  # balance the pos and neg actions, following RPIN
    # if we use more than 10% of data, we cannot keep the pos-neg ratio
    # since there are much less positive actions in the dataset
    # we discovered that when using white background, SAVi has difficulty
    # segmenting light-color objects, so we make the background black
    reverse_color = True

    # our preliminary experiments show that batch size 32 is better than 64,
    # which is the number used for other SAVi models
    # so we stick to this value for all folds
    # in Slot-Attention, randomness is necessary to trigger scene decomposition
    # so using a larger batch size may reduce such randomness in the gradients
    train_batch_size = 32 // gpus
    val_batch_size = int(train_batch_size * 1.5)  # *2 causes OOM, weird...
    num_workers = 8

    # model configs
    model = 'StoSAVi'  # we actually use the deterministic version here
    resolution = (128, 128)
    input_frames = n_sample_frames

    # Slot Attention
    slot_dict = dict(
        num_slots=8,
        slot_size=128,
        slot_mlp_size=256,
        num_iterations=2,
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
        dec_resolution=(16, 16),  # larger size to better capture small objects
        dec_ks=5,
        dec_norm='',
    )

    # Predictor
    pred_dict = dict(
        pred_type='transformer',
        pred_rnn=True,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=slot_dict['slot_size'] * 4,
        pred_sg_every=None,
    )

    # loss configs
    loss_dict = dict(
        use_post_recon_loss=True,
        kld_method='none',  # standard SAVi
    )

    post_recon_loss_w = 1.  # posterior slots image recon
    kld_loss_w = 1e-4  # kld on kernels distribution
