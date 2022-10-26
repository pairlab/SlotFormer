from .readout import PhysionReadout


def build_model(params):
    """Build slot-based VQA model Aloe."""
    assert params.model == 'PhysionReadout', f'Unknown model: {params.model}'
    model = PhysionReadout(readout_dict=params.readout_dict)
    return model
