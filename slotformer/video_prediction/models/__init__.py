from .slotformer import SlotFormer
from .single_step_slotformer import SingleStepSlotFormer
from .steve_slotformer import STEVESlotFormer


def build_model(params):
    if params.model == 'SlotFormer':
        return SlotFormer(
            resolution=params.resolution,
            clip_len=params.input_frames,
            slot_dict=params.slot_dict,
            dec_dict=params.dec_dict,
            rollout_dict=params.rollout_dict,
            loss_dict=params.loss_dict,
        )
    elif params.model == 'SingleStepSlotFormer':
        return SingleStepSlotFormer(
            resolution=params.resolution,
            clip_len=params.input_frames,
            slot_dict=params.slot_dict,
            dec_dict=params.dec_dict,
            rollout_dict=params.rollout_dict,
            loss_dict=params.loss_dict,
        )
    elif params.model == 'STEVESlotFormer':
        return STEVESlotFormer(
            resolution=params.resolution,
            clip_len=params.input_frames,
            slot_dict=params.slot_dict,
            dvae_dict=params.dvae_dict,
            dec_dict=params.dec_dict,
            rollout_dict=params.rollout_dict,
            loss_dict=params.loss_dict,
        )
    else:
        raise NotImplementedError(f'{params.model} is not implemented.')
