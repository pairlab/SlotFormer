from .clevrer_data import clevrer_collate_fn, build_clevrer_slots_vqa_dataset


def build_dataset(params, test_set=False):
    assert params.dataset == 'clevrer_slots'
    return build_clevrer_slots_vqa_dataset(
        params, test_set=test_set), clevrer_collate_fn
