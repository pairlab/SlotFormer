from .savi_model import StoSAViModel
from .dVAE import dVAE
from .steve_model import STEVEModel
from .utils import get_lr
from .steve_utils import cosine_anneal, gumbel_softmax, make_one_hot


def build_model(params):
    if params.model == 'StoSAVi':
        return StoSAViModel(
            resolution=params.resolution,
            clip_len=params.input_frames,
            slot_dict=params.slot_dict,
            enc_dict=params.enc_dict,
            dec_dict=params.dec_dict,
            pred_dict=params.pred_dict,
            loss_dict=params.loss_dict,
        )
    elif params.model == 'dVAE':
        return dVAE(vocab_size=params.vocab_size, img_channels=3)
    elif params.model == 'STEVE':
        return STEVEModel(
            resolution=params.resolution,
            clip_len=params.input_frames,
            slot_dict=params.slot_dict,
            dvae_dict=params.dvae_dict,
            enc_dict=params.enc_dict,
            dec_dict=params.dec_dict,
            pred_dict=params.pred_dict,
        )
    else:
        raise NotImplementedError(f'{params.model} is not implemented.')
