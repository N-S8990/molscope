"""Build the exploratory GPR6 motif panel on the verified 12-replica window.

Selections refer to topology residue IDs checked against the published GPR6
motif assignments. This panel was designed after the 39-feature pilot and must
not be presented as preregistered or as independent confirmation of that run.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "protein_first_100ns_1000_frames/project.json"
EXPANDED = ROOT / "protein_first_100ns_1000_frames/project_expanded.json"

# Grouped by receptor opening, toggle/core, sodium-site geometry, DRY/NPxxY,
# and pocket geometry. These are Cα distances, not direct interaction counts.
CA_PAIRS = [
    [166, 274], [166, 276], [167, 274],
    [152, 292], [152, 234], [234, 292],
    [156, 237], [156, 288], [237, 288],
    [118, 155], [118, 321], [155, 321], [118, 325], [321, 325],
    [165, 329], [166, 329], [167, 329], [274, 329], [274, 325],
    [128, 152], [152, 295], [152, 315], [292, 315], [234, 288], [237, 292],
]

# Side-chain rotamers use circular sin/cos encodings; each atom quartet is
# checked against the deposited topology by the plugin at extraction time.
CHI_RESIDUES = [118, 152, 155, 156, 166, 288, 292, 321, 329]


def main() -> None:
    project = json.loads(BASE.read_text())
    if len(project["systems"][0]["ensembles"]) != 2 or len(CA_PAIRS) != 25:
        raise ValueError("Unexpected GPR6 base project or pair panel")
    project["project_id"] = "gpr6_public_3h_vs_no_added_expanded_motifs_first_100ns"
    project["analysis"]["plugins"] = [
        "protein.ca_distances_contacts",
        "protein.sidechain_torsions",
        "protein.secondary_structure_dssp",
    ]
    project["analysis"]["plugin_parameters"] = {
        "protein.ca_distances_contacts": {
            "residue_pairs": CA_PAIRS,
            "include_contacts": False,
            "max_pairs": len(CA_PAIRS),
            "minimum_sequence_separation": 3,
        },
        "protein.sidechain_torsions": {
            "residue_ids": CHI_RESIDUES,
            "include_chi2": True,
            "max_torsions": 32,
        },
        "protein.secondary_structure_dssp": {"per_residue": False},
    }
    EXPANDED.write_text(json.dumps(project, indent=2) + "\n")
    print(f"Prepared exploratory expanded project: {EXPANDED}")


if __name__ == "__main__":
    main()
