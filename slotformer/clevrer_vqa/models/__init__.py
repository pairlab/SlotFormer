from nerv.utils import load_obj

from .aloe import CLEVRERAloe
from .transformer import CLEVRERTransformerModel


def build_transformer(params):
    """Build VQA Transformer model."""
    vocab = load_obj(params.vocab_file)
    lang_dict = dict(
        question_vocab_size=len(vocab['q_vocab']),
        answer_vocab_size=len(vocab['a_vocab']),
        question_len=params.max_question_len,
    )
    transformer = CLEVRERTransformerModel(
        transformer_dict=params.transformer_dict,
        lang_dict=lang_dict,
        vision_dict=params.vision_dict,
        loss_dict=params.loss_dict,
    )
    return transformer


def build_model(params):
    """Build slot-based VQA model Aloe."""
    assert params.model == 'CLEVRERAloe', f'Unknown model: {params.model}'
    transformer = build_transformer(params)
    model = CLEVRERAloe(transformer_model=transformer)
    return model
