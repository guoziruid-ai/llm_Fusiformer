from mp_api.client import MPRester
from robocrys.condense.condenser import StructureCondenser
from robocrys import StructureDescriber
import json

#G 硬编码（建议修改）
API_KEY = "iRQn8R8avlT2r0wyKviRYKHny9ztTVtp"
mpr = MPRester(API_KEY)

material_ids = ["mp-10"]

results = mpr.materials.summary.search(
    material_ids=material_ids,
    fields=[
        "material_id",
        "formula_pretty",
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "total_magnetization",
        "bulk_modulus",
        "shear_modulus",
        "structure"
    ]
)

condenser = StructureCondenser()
describer = StructureDescriber()

dataset = []

for r in results:
    struct = r.structure

    condensed = condenser.condense_structure(struct)
    description = describer.describe(condensed)

    entry = {
        "mp_id": r.material_id,
        "formula": r.formula_pretty,

        "e_hull": r.energy_above_hull,
        "e_form": r.formation_energy_per_atom,
        "gap_pbe": r.band_gap,
        "mu_b": r.total_magnetization,
        "bulk_modulus": r.bulk_modulus,
        "shear_modulus": r.shear_modulus,

        "description": description,

        "atoms": {
            "lattice_mat": struct.lattice.matrix.tolist(),
            "coords": [s.frac_coords.tolist() for s in struct.sites],
            "elements": [str(s.specie) for s in struct.sites],
            "abc": list(struct.lattice.abc),
            "angles": list(struct.lattice.angles),
            "cartesian": False,
            "props": [""] * len(struct.sites)
        }
    }

    dataset.append(entry)

with open("my_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("已保存my_dataset.json")
