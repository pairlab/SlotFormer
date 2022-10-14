from .slotformer import SlotFormer
from .single_step_slotformer import SingleStepSlotFormer
from .steve_slotformer import STEVESlotFormer


def build_model(params):
    assert params.model in [
        'SlotFormer', 'SingleStepSlotFormer', 'STEVESlotFormer'
    ], f'{params.model} is not implemented.'
    return eval(params.model)(
        resolution=params.resolution,
        clip_len=params.input_frames,
        slot_dict=params.slot_dict,
        dec_dict=params.dec_dict,
        rollout_dict=params.rollout_dict,
        loss_dict=params.loss_dict,
    )
