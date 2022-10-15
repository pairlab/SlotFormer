from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 2  # 1 GPU should also be good
    max_epochs = 20  # ~700k steps
    save_interval = 0.25  # save every 0.25 epoch
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 8  # Physion has 8 scenarios

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-3
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay, no gradient clipping

    # data settings
    dataset = 'physion_training'
    data_root = './data/Physion'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 1  # train on single frames
    frame_offset = 1  # no offset
    video_len = 150  # take the first 150 frames of each video
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'dVAE'
    resolution = (128, 128)
    vocab_size = 4096  # codebook size

    # temperature for gumbel softmax
    # decay from 1.0 to 0.1 in the first 15% of total steps
    init_tau = 1.
    final_tau = 0.1
    tau_decay_pct = 0.15

    # loss settings
    recon_loss_w = 1.
