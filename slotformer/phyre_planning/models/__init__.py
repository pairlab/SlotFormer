from .readout import PHYREReadout


def build_model(params):
    """Build slot-based VQA model Aloe."""
    assert params.model == 'PHYREReadout', f'Unknown model: {params.model}'
    model = PHYREReadout(readout_dict=params.readout_dict)
    return model
