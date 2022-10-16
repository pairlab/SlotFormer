from .clevrer import clevrer_collate_fn, build_clevrer_slots_vqa_dataset


def build_dataset(params, test_set=False):
    assert params.dataset == 'clevrer_slots'
    if test_set:
        test_dataset = build_clevrer_slots_vqa_dataset(params, test_set=True)
        return test_dataset, clevrer_collate_fn
    train_dataset, val_dataset = build_clevrer_slots_vqa_dataset(
        params, test_set=False)
    return train_dataset, val_dataset, clevrer_collate_fn
