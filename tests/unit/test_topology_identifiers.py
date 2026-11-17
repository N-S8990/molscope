import MDAnalysis as mda

from molscope.features.base import _chain_or_segment_id
from molscope.features.geometry.generic import _residue_feature_id


class SegmentOnlyTopology:
    @property
    def chainIDs(self):
        raise mda.exceptions.NoDataError('chainID unavailable')

    @property
    def segids(self):
        return ['ADK']


def test_segment_identity_is_used_when_topology_has_no_chain_ids():
    atoms = SegmentOnlyTopology()
    assert _chain_or_segment_id(atoms) == 'segment_ADK'


def test_residue_feature_names_use_segment_fallback():
    residue = type('Residue', (), {'atoms': SegmentOnlyTopology(), 'resname': 'ALA', 'resid': 42})()
    assert _residue_feature_id(residue) == 'segment_ADK__ALA_42'
