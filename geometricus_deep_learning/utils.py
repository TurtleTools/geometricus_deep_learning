import typing as ty
import numpy as np
from dataclasses import dataclass
from geometricus import MomentType, MomentInvariants, SplitType
from torch_geometric.data import DataLoader, Data
import networkx as nx
from glob import glob
from torch_geometric import utils as torch_geo_utils
import torch
from random import shuffle
from sklearn.preprocessing import StandardScaler
from biotransformers import BioTransformers
import prody as pd


@dataclass
class SeqData:
    seq: str
    x: torch.Tensor
    pdb_id: str


def transform_pdbseqs(pdb_folder) -> ty.Dict[str, SeqData]:
    bio_trans = BioTransformers(backend="protbert")
    files = list(glob(pdb_folder + "/*"))
    pdbs = [pd.parsePDB(x) for x in files]
    sequences = [x.select('protein and name CA').getSequence() for x in pdbs]
    ids = [x.split("/")[-1] for x in files]
    sequence_embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls', 'mean'))
    return {ids[i]: SeqData(sequences[i],
                            sequence_embeddings[i],
                            ids[i]) for i in range(len(ids))}


@dataclass
class InvariantType:
    type: SplitType
    k: int
    moment_types: ty.Union[ty.List[MomentType], None] = None


@dataclass
class MomentInvariantsSavable:
    moments: ty.Union[np.ndarray, None]
    coordinates: ty.Union[np.ndarray, None]

    @classmethod
    def from_invariant(cls, invariant: MomentInvariants):
        return cls(invariant.moments, invariant.coordinates)


def invariants_from_pdb_folder(pdb_file_path: str,
                               invariant_type: InvariantType,
                               ids_to_use: ty.Union[ty.Set[str], None] = None) -> ty.Dict[
    str, MomentInvariants]:
    files = glob(pdb_file_path + "*" if pdb_file_path.endswith("/") else pdb_file_path + "/*")
    folder_invariants: ty.Dict[str, MomentInvariants] = dict()
    moment_types = (
        invariant_type.moment_types
        if invariant_type.moment_types is not None
        else [
            MomentType.O_3,
            MomentType.F,
            MomentType.O_5,
            MomentType.O_4,
            MomentType.phi_10,
            MomentType.phi_11,
            MomentType.phi_12,
            MomentType.phi_13,
            MomentType.phi_2,
            MomentType.phi_3,
            MomentType.phi_4,
            MomentType.phi_5,
            MomentType.phi_6,
            MomentType.phi_7,
            MomentType.phi_8,
            MomentType.phi_9,
        ]
    )
    if ids_to_use is not None:
        files = [x for x in files if x.split("/")[-1] in ids_to_use]
    for filename in files:
        try:
            folder_invariants[filename.split("/")[-1]] = MomentInvariants.from_pdb_file(filename,
                                                                                        moment_types=moment_types,
                                                                                        split_size=invariant_type.k,
                                                                                        split_type=invariant_type.type)
        except:  # TODO handle bad pdbs
            continue

    scaler = StandardScaler()
    longest_invariant = max(list(folder_invariants.values()), key=lambda x: x.moments.shape[0])
    scaler.fit(longest_invariant.moments)
    for invariant in folder_invariants.values():
        invariant.moments = scaler.transform(invariant.moments)

    return folder_invariants


def concat_invariants(invariant1: ty.Union[MomentInvariants, MomentInvariantsSavable],
                      invariant2: ty.Union[MomentInvariants, MomentInvariantsSavable]) -> MomentInvariants:
    if invariant1.moments is None and invariant2.moments is None:
        return invariant1
    elif invariant1.moments is None:
        return invariant2
    elif invariant2.moments is None:
        return invariant1
    else:
        assert invariant1.coordinates.shape == invariant2.coordinates.shape
        invariant1.moments = np.hstack((invariant1.moments, invariant2.moments))
        return invariant1


def invariant_to_graph(invariant: ty.Union[MomentInvariants, MomentInvariantsSavable], lim: float = 6.) -> nx.DiGraph:
    from sklearn.metrics import euclidean_distances
    distances = euclidean_distances(invariant.coordinates)
    prot_moments = invariant.moments
    graph = nx.DiGraph()
    graph.add_node(0, x=prot_moments[0].astype("float32"))
    for i in range(1, prot_moments.shape[0]):
        graph.add_node(i, x=prot_moments[i].astype("float32"))
        graph.add_edge(i - 1, i)
    for i in range(distances.shape[0]):
        current = distances[i]
        for j in np.where(current < lim)[0]:
            if i != j and j != (i - 1) and j != (i + 1):
                graph.add_edge(i, j)
    return graph


def dataloader_from_structure_and_sequence(structure_dataloaders: ty.Dict[str, np.ndarray],
                                           sequence_data: ty.Dict[str, SeqData],
                                           train_ratio: float = 0.8,
                                           batch_no: int = 512) -> ty.Tuple[np.ndarray, DataLoader]:

    pairs: ty.List[ty.Tuple[np.ndarray, SeqData]] = [(structure_dataloaders[k],
                                                      sequence_data[k]) for k in structure_dataloaders.keys()]
    train_dataset = pairs[:int(train_ratio * len(pairs))]
    test_dataset = pairs[int(train_ratio * len(pairs)):]
    return (DataLoader(train_dataset, batch_size=batch_no, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_no, shuffle=False))


def transform_geometricus_dataset_for_training(filename_to_classname: ty.Dict[str, str],
                                               invariants: ty.Dict[
                                                   str, ty.Union[MomentInvariants, MomentInvariantsSavable]],
                                               train_ratio: float = 0.8,
                                               graph_distance_threshold: float = 6.,
                                               batch_no: int = 1028) -> ty.Tuple[DataLoader, DataLoader, dict]:
    train_ratio = train_ratio if train_ratio <= 1 else 1.
    filename_to_classname = {k: v for k, v in filename_to_classname.items() if k in invariants}
    all_keys = list(filename_to_classname.keys())
    shuffle(all_keys)

    all_graphs = [invariant_to_graph(invariants[x], graph_distance_threshold) for x in all_keys]

    ys = [filename_to_classname[x] for x in all_keys]
    xs = [torch_geo_utils.from_networkx(x) for x in all_graphs]

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit([[x] for x in set(ys)])

    id_to_classname = {encoder.transform([[k]]).argmax(): k for k in ys}

    for i, data in enumerate(xs):
        data.y = torch.from_numpy(np.array([encoder.transform([[ys[i]]]).toarray().astype("int32")[0].argmax()])).type(
            torch.LongTensor)
        data.x = data.x.type(torch.float32)
        data.pdb_id = all_keys[i]
    train_dataset = xs[:int(len(xs) * train_ratio)]
    test_dataset = xs[int(len(xs) * train_ratio):]

    return (DataLoader(train_dataset, batch_size=batch_no, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_no, shuffle=False), id_to_classname)
