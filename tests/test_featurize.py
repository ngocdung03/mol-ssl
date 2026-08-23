from src.featurize import ATOM_FEAT_DIM, BOND_FEAT_DIM, mol_to_data


def test_aspirin_graph_shape():
    data = mol_to_data("CC(=O)Oc1ccccc1C(=O)O")
    assert data is not None
    assert data.x.shape == (13, ATOM_FEAT_DIM)
    assert data.edge_index.shape[0] == 2
    # undirected: 2 directed edges per bond, 13 bonds in aspirin
    assert data.edge_index.shape[1] == 26
    assert data.edge_attr.shape == (26, BOND_FEAT_DIM)


def test_single_atom_molecule_has_no_edges():
    data = mol_to_data("C")
    assert data is not None
    assert data.x.shape == (1, ATOM_FEAT_DIM)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, BOND_FEAT_DIM)


def test_unparseable_smiles_returns_none():
    assert mol_to_data("not_a_molecule[[[") is None


def test_label_is_attached_as_row():
    data = mol_to_data("CCO", y=[1.0, 0.0])
    assert data.y.shape == (1, 2)
