import os

from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 2
    max_epochs = 400  # 240k steps
    eval_interval = 20  # evaluate every 20 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 5  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    lr = 1e-3
    warmup_steps_pct = 0.1  # warmup in the first 10% of total steps
    # no weight decay, no gradient clipping

    # data settings
    dataset = 'clevrer_slots'
    data_root = './data/CLEVRER'
    slots_root = './data/CLEVRER/clevrer_slots.pkl'
    # put absolute path here
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_file = os.path.join(cur_dir, '../datasets/cache/CLEVRER_vocab.json')
    n_sample_frames = 25  # load 25 slots for VQA
    slot_size = 128
    max_n_objects = 6
    max_question_len = 20
    max_choice_len = 12
    shuffle_obj = False  # SAVi has temporal consistency
    train_batch_size = 256 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8

    # model configs
    model = 'CLEVRERAloe'
    transformer_dict = dict(
        input_len=(max_n_objects + 1) * n_sample_frames + max_question_len +
        max_choice_len,
        input_dim=16,
        pos_enc='learnable',
        num_layers=12,
        num_heads=8,
        ffn_dim=512,
        norm_first=True,
        cls_mlp_size=128,
    )
    vision_dict = dict(vision_dim=slot_size, )

    # loss configs
    loss_dict = dict(use_mask_obj_loss=False, )

    cls_answer_loss_w = 1.
    mc_answer_loss_w = 1.
    mask_obj_loss_w = 0.01
