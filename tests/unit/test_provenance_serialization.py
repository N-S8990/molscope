import json
import numpy as np
from molscope.contracts import FeatureProvenance


def test_numpy_residue_indices_are_json_serializable():
    provenance = FeatureProvenance("dssp", "1", {"residue_mapping": [{"resindex": np.int64(3)}]})
    assert json.loads(json.dumps(provenance.to_dict()))["parameters"]["residue_mapping"][0]["resindex"] == 3
