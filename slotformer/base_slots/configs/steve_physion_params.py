from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    # STEVE on 128x128 resolution is very memory-consuming
    # the setting here is for training on 4 RTX6000 GPUs with 24GB memory each
    gpus = 4
    max_epochs = 10  # ~460k steps
    save_interval = 0.05  # training is very slow, save every 0.05 epoch
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 8  # Physion has 8 scenarios

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4  # 1e-4 for the main STEVE model
    dec_lr = 3e-4  # 3e-4 for the Transformer decoder
    clip_grad = 0.05  # following the paper
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps

    # data settings
    dataset = 'physion_training'
    data_root = './data/Physion'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 6  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 150  # take the first 150 frames of each video
    train_batch_size = 48 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'STEVE'
    resolution = (128, 128)
    input_frames = n_sample_frames

    # Slot Attention
    slot_size = 192
    slot_dict = dict(
        # the objects are harder to define in Physion than in e.g. CLEVRER
        # e.g. should a stack of 6 boxes be considered as 1 or 6 objects?
        #      among these boxes, some move together, some usually fall, etc.
        # we don't ablate this `num_slots`, so we are not sure if it will make
        # a huge difference to the results
        # qualitatively, 6 achieves a reasonable scene decomposition result
        num_slots=6,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=2,
    )

    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=4096,
        dvae_ckp_path='pretrained/dvae_physion_params/model_20.pth',
    )

    # CNN Encoder
    enc_dict = dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=slot_size,
        enc_norm='',
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
    )

    # Predictor
    pred_dict = dict(
        pred_type='transformer',
        pred_rnn=True,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=slot_size * 4,
        pred_sg_every=None,
    )

    # loss settings
    loss_dict = dict(
        use_img_recon_loss=False,  # additional img recon loss via dVAE decoder
    )

    token_recon_loss_w = 1.
    img_recon_loss_w = 1.
