import pytest
import torch
from warping_server.dataset_loader import load_person_inputs

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")

def test_valid_person_id():
    person_id = "00826_00"
    person, agnostic, densepose, pose, parse_agnostic = \
        load_person_inputs(DATASET_ROOT, person_id)

    assert isinstance(person, torch.Tensor)
    assert person.shape[1:] == (256, 192)

def test_invalid_person_id():
    with pytest.raises(FileNotFoundError):
        load_person_inputs(DATASET_ROOT, "invalid_000")