from src.splits import heldout_scaffolds, load_split, save_split, scaffold_smiles, scaffold_split

# two benzene-scaffold molecules, two naphthalene, one acyclic
SMILES = ["c1ccccc1C", "c1ccccc1CC", "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C", "CCCCO"]


def test_scaffold_smiles_groups_substituted_benzenes():
    assert scaffold_smiles("c1ccccc1C") == scaffold_smiles("c1ccccc1CC")


def test_scaffold_split_puts_no_scaffold_in_two_partitions():
    split = scaffold_split(SMILES, 0.6, 0.2, 0.2)
    seen: dict[str, str] = {}
    for part, idxs in split.items():
        for i in idxs:
            scaf = scaffold_smiles(SMILES[i])
            assert seen.setdefault(scaf, part) == part, f"scaffold {scaf} in two partitions"


def test_scaffold_split_is_deterministic():
    assert scaffold_split(SMILES) == scaffold_split(SMILES)


def test_manifest_roundtrip_and_heldout(tmp_path):
    split = scaffold_split(SMILES, 0.6, 0.2, 0.2)
    path = tmp_path / "toy.json"
    save_split(path, "toy", split, SMILES)
    manifest = load_split(path)
    assert manifest["split_type"] == "scaffold_bemis_murcko"
    assert manifest["indices"] == split
    held = heldout_scaffolds(manifest)
    train_scafs = set(manifest["scaffolds"]["train"])
    assert not (held & train_scafs)


def test_load_split_refuses_random_manifest(tmp_path):
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"split_type": "random", "indices": {}, "scaffolds": {}}))
    try:
        load_split(path)
    except ValueError:
        return
    raise AssertionError("load_split accepted a random split manifest")
