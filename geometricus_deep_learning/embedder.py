import typing as ty
import numpy as np
from dataclasses import dataclass
from geometricus import MomentType, MomentInvariants, SplitType
from torch_geometric.data import DataLoader
import networkx as nx
from glob import glob
from torch_geometric import utils
import torch
from geometricus_deep_learning.geometric_models import GCN
from random import shuffle
from sklearn.preprocessing import StandardScaler
from geometricus_deep_learning import utils


@dataclass
class MomentInvariantsSavable:
    moments: np.ndarray
    coordinates: np.ndarray

    @classmethod
    def from_invariant(cls, invariant: MomentInvariants):
        return cls(invariant.moments, invariant.coordinates)


def invariants_from_pdb_folder(pdb_file_path: str,
                               split_type: SplitType = SplitType.KMER,
                               size: int = 30,
                               invariant_types: ty.Union[ty.List[MomentType], None] = None) -> ty.Dict[
    str, MomentInvariants]:
    files = glob(pdb_file_path + "*" if pdb_file_path.endswith("/") else pdb_file_path + "/*")
    folder_invariants: ty.Dict[str, MomentInvariants] = dict()
    invariant_types = (
        invariant_types
        if invariant_types is not None
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

    for filename in files:
        try:
            folder_invariants[filename.split("/")[-1]] = MomentInvariants.from_pdb_file(filename,
                                                                                 moment_types=invariant_types,
                                                                                 split_size=size,
                                                                                 split_type=split_type)
        except:
            continue

    scaler = StandardScaler()
    longest_invariant = max(list(folder_invariants.values()), key=lambda x: x.moments.shape[0])
    scaler.fit(longest_invariant.moments)
    for invariant in folder_invariants.values():
        invariant.moments = scaler.transform(invariant.moments)

    return folder_invariants


def concat_invariants(invariant1: ty.Union[MomentInvariants, MomentInvariantsSavable],
                      invariant2: ty.Union[MomentInvariants, MomentInvariantsSavable]) -> MomentInvariants:
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


def transform_geometricus_dataset_for_training(filename_to_classname: ty.Dict[str, str],
                                               invariants: ty.Dict[
                                                   str, ty.Union[MomentInvariants, MomentInvariantsSavable]],
                                               train_ratio: float = 0.8,
                                               graph_distance_threshold: float = 6.,
                                               batch_no: int = 1028) -> ty.Tuple[DataLoader, DataLoader]:
    train_ratio = train_ratio if train_ratio <= 1 else 1.
    all_keys = list(filename_to_classname.keys())
    shuffle(all_keys)
    all_keys = list(all_keys)

    all_graphs = [invariant_to_graph(invariants[x], graph_distance_threshold) for x in all_keys]

    ys = [filename_to_classname[x] for x in all_keys]
    xs = [utils.from_networkx(x) for x in all_graphs]

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit([[x] for x in set(ys)])

    for i, data in enumerate(xs):
        data.y = torch.from_numpy(np.array([encoder.transform([[ys[i]]]).toarray().astype("int32")[0].argmax()])).type(
            torch.LongTensor)
        data.x = data.x.type(torch.float32)
    train_dataset = xs[:int(len(xs) * train_ratio)]
    test_dataset = xs[int(len(xs) * train_ratio):]

    return (DataLoader(train_dataset, batch_size=batch_no, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_no, shuffle=False))


def transform_geometricus_dataset_for_training_dual_class(filename_to_classname1: ty.Dict[str, str],
                                                          filename_to_classname2: ty.Dict[str, str],
                                                          invariants: ty.Dict[
                                                              str, ty.Union[MomentInvariants, MomentInvariantsSavable]],
                                                          train_ratio: float = 0.8,
                                                          graph_distance_threshold: float = 6.,
                                                          batch_no: int = 1028) -> ty.Tuple[DataLoader, DataLoader]:
    train_ratio = train_ratio if train_ratio <= 1 else 1.
    all_keys = list(filename_to_classname1.keys())
    shuffle(all_keys)
    all_keys = list(all_keys)

    all_graphs = [invariant_to_graph(invariants[x], graph_distance_threshold) for x in all_keys]

    ys1 = [filename_to_classname1[x] for x in all_keys]
    ys2 = [filename_to_classname2[x] for x in all_keys]
    xs = [utils.from_networkx(x) for x in all_graphs]

    from sklearn.preprocessing import OneHotEncoder
    encoder1 = OneHotEncoder(handle_unknown="ignore")
    encoder1.fit([[x] for x in set(ys1)])
    encoder2 = OneHotEncoder(handle_unknown="ignore")
    encoder2.fit([[x] for x in set(ys2)])

    for i, data in enumerate(xs):
        y = np.array([encoder1.transform([[ys1[i]]]).toarray().astype("int32")[0].argmax(),
                      encoder2.transform([[ys2[i]]]).toarray().astype("int32")[0].argmax()])
        data.y = torch.from_numpy(y).type(torch.LongTensor)
        data.x = data.x.type(torch.float32)
    train_dataset = xs[:int(len(xs) * train_ratio)]
    test_dataset = xs[int(len(xs) * train_ratio):]

    return (DataLoader(train_dataset, batch_size=batch_no, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_no, shuffle=False))


def train_model(train_dataset,
                test_dataset,
                number_of_node_features,
                hidden_channels,
                number_of_classes,
                aggregator="mean",
                epochs: int = 5_000,
                lr=0.0005):
    model = GCN(number_of_node_features, hidden_channels, number_of_classes, aggregator=aggregator).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        for data in train_dataset:  # Iterate in batches over the training dataset.
            data = data.cuda()
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.cuda()
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    train_acc = test(train_dataset)
    test_acc = test(test_dataset)
    print(f'Initial: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    tests = list()
    trains = list()
    for epoch in range(1, epochs):
        train()
        train_acc = test(train_dataset)
        test_acc = test(test_dataset)
        trains.append(train_acc)
        tests.append(test_acc)
        if (epoch % 10) == 0:
            print(f'Epoch: {epoch:03d}, Train Acc: {np.mean(trains):.4f}, Test Acc: {np.mean(tests):.4f}')
            trains = list()
            tests = list()
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    return model


def test():
    pdb_folder = "../cath_data/dompdb/"
    invariants1 = invariants_from_pdb_folder(pdb_folder, split_type=SplitType.KMER,
                                             size=30)

    invariants2 = invariants_from_pdb_folder(pdb_folder, split_type=SplitType.RADIUS,
                                             size=10)
    invariants = {k: concat_invariants(v, invariants2[k]) for k, v in invariants1.items()}
    single_invariant = invariants[list(invariants.keys())[0]]
    print("invariants extracted")

    domain_info = utils.DomainInfo.from_domainlist_file("../cath_data/cath-domain-list-S100.txt").domains
    cath_mapping = {k: f"{v.c_class}-{v.architecture}-{v.topology}" for k, v in domain_info.items() if k in invariants}

    keys, counts = np.unique(list(cath_mapping.values()), return_counts=True)
    least_allowed_class_count = 400
    keys_to_use = {x for x in keys[np.where(counts >= least_allowed_class_count)[0]]}
    cath_mapping = {k: v for k, v in cath_mapping.items() if v in keys_to_use}
    print("cath info linked to files")

    train_data, test_data = transform_geometricus_dataset_for_training(cath_mapping,
                                                                       invariants)
    train_model(
        train_data,
        test_data,
        number_of_node_features=single_invariant.moments.shape[1],
        hidden_channels=256,
        number_of_classes=len(keys_to_use),
        lr=0.0001
    )


if __name__ == "__main__":
    test()
