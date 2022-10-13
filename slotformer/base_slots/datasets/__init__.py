from .obj3d import build_obj3d_dataset, build_obj3d_slots_dataset
from .clevrer import build_clevrer_dataset, build_clevrer_slots_dataset
from .physion import build_physion_dataset, build_physion_slots_dataset, \
    build_physion_slots_label_dataset
from .phyre import build_phyre_dataset, build_phyre_slots_dataset, \
    build_phyre_rollout_slots_dataset


def build_dataset(params, val_only=False):
    if 'physion' not in params.dataset:
        return eval(f'build_{params.dataset}_dataset')(
            params, val_only=val_only)
    # physion dataset has different subsets
    if 'slots_label' in params.dataset:
        return build_physion_slots_label_dataset(params, val_only=val_only)
    elif 'slots' in params.dataset:
        return build_physion_slots_dataset(params, val_only=val_only)
    elif 'physion' in params.dataset:
        return build_physion_dataset(params, val_only=val_only)
    else:
        raise NotImplementedError(f'Dataset {params.dataset} is not supported')
