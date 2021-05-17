import typing as ty
import numpy as np
from dataclasses import dataclass
from geometricus import MomentType, MomentInvariants, SplitType
from torch_geometric.data import DataLoader
import networkx as nx
from glob import glob
from torch_geometric import utils as torch_geo_utils
import torch
from geometricus_deep_learning.geometric_models import GCN
from random import shuffle
from sklearn.preprocessing import StandardScaler
from geometricus_deep_learning import utils
import umap
from pathlib import Path
import pickle


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
                               invariant_type: InvariantType) -> ty.Dict[
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

    for filename in files:
        try:
            folder_invariants[filename.split("/")[-1]] = MomentInvariants.from_pdb_file(filename,
                                                                                        moment_types=moment_types,
                                                                                        split_size=invariant_type.k,
                                                                                        split_type=invariant_type.type)
        except: # TODO handle bad pdbs
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


def transform_geometricus_dataset_for_training(filename_to_classname: ty.Dict[str, str],
                                               invariants: ty.Dict[
                                                   str, ty.Union[MomentInvariants, MomentInvariantsSavable]],
                                               train_ratio: float = 0.8,
                                               graph_distance_threshold: float = 6.,
                                               batch_no: int = 1028) -> ty.Tuple[DataLoader, DataLoader]:
    train_ratio = train_ratio if train_ratio <= 1 else 1.
    all_keys = list(filename_to_classname.keys())
    shuffle(all_keys)

    all_graphs = [invariant_to_graph(invariants[x], graph_distance_threshold) for x in all_keys]

    ys = [filename_to_classname[x] for x in all_keys]
    xs = [torch_geo_utils.from_networkx(x) for x in all_graphs]

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit([[x] for x in set(ys)])

    for i, data in enumerate(xs):
        data.y = torch.from_numpy(np.array([encoder.transform([[ys[i]]]).toarray().astype("int32")[0].argmax()])).type(
            torch.LongTensor)
        data.x = data.x.type(torch.float32)
        data.pdb_id = all_keys[i]
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
    return model, (train_acc, test_acc)


def test():
    pdb_folder = "../data/cath/"
    invariants = GeometricusGraphEmbedder.get_multi_invariants(pdb_folder, [
        InvariantType(SplitType.KMER, 30),
        InvariantType(SplitType.RADIUS, 10),
    ])
    single_invariant = invariants[list(invariants.keys())[0]]
    print("invariants extracted")
    domain_info = utils.DomainInfo.from_domainlist_file("../data/cath-domain-list-S100.txt").domains
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


@dataclass
class EmbedderMeta:
    model_path: str
    umap_transformer_path: str
    pdb_folder: str
    self_path: str
    classes_to_ids: ty.Dict[str, int]
    invariant_types: ty.List[InvariantType]
    train_acc: float
    test_acc: float
    original_invariants_file: str


@dataclass
class GeometricusGraphEmbedder:
    ids: ty.Union[ty.List[str], None]
    model_meta: ty.Union[EmbedderMeta, None] = None
    model: ty.Union[GCN, None] = None
    pdb_target_folder: ty.Union[str, None] = None
    pdb_training_folder: ty.Union[str, None] = None
    umap_transformer: ty.Union[str, umap.UMAP, None] = None

    @staticmethod
    def get_multi_invariants(pdb_file_path: str,
                             invariant_types: ty.List[InvariantType]) -> ty.Dict[str, MomentInvariantsSavable]:
        invariant_collection: ty.List[ty.Dict[str, MomentInvariantsSavable]] = list()
        for invariant_type in invariant_types:
            invariant_collection.append(
                {k: MomentInvariantsSavable.from_invariant(v) for k, v in
                 invariants_from_pdb_folder(pdb_file_path, invariant_type).items()}
            )
        # 2. concat. different types together into single array of moments
        invariant_all: ty.Dict[str, MomentInvariantsSavable] = {k: MomentInvariantsSavable(None, None) for k in
                                                                invariant_collection[0].keys()}

        for invariant in invariant_collection:
            invariant_all = {k: concat_invariants(invariant_all[k], invariant[k]) for k in invariant_all.keys()}
        return invariant_all

    @staticmethod
    def get_embedding(loaders, model) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        res = list()
        labels = []
        predicted_labels = []
        pdb_ids = []
        for loader in loaders:
            for data in loader:
                data = data.cuda()
                out = model(data.x, data.edge_index, data.batch)
                res.append(out.cpu().detach().numpy())
                labels.append(data.y.cpu().detach().numpy())
                predicted_labels.append(out.argmax(dim=1).cpu().detach().numpy())
                pdb_ids.append(data.pdb_id)
        return np.concatenate(res), np.concatenate(labels), np.concatenate(predicted_labels), np.concatenate(pdb_ids)

    @classmethod
    def from_training(cls,
                      pdb_file_path: str,
                      invariant_types: ty.List[InvariantType],
                      pdb_file_to_class_mapping: ty.Dict[str, str],
                      hidden_channels: int = 128,
                      learning_rate: float = 0.001,
                      file_output_path: str = "./data/models/",
                      embedding_size: int = 10,
                      epochs: int = 1_000) -> "GeometricusGraphEmbedder":

        # 1. create invariants according to given invariant types
        # 2. concat. different types together into single array of moments
        invariant_all = cls.get_multi_invariants(pdb_file_path, invariant_types)
        single_invariant = invariant_all[list(invariant_all.keys())[0]]
        # 3. Remove mappings which are not included
        pdb_file_to_class_mapping = {k: v for k, v in pdb_file_to_class_mapping.items() if k in invariant_all}

        # 4. Split into train and test sets
        train_data, test_data = transform_geometricus_dataset_for_training(pdb_file_to_class_mapping, invariant_all)

        # 5. train model and store relevant metadata
        model, (train_acc, test_acc) = train_model(train_data, test_data,
                                                   number_of_node_features=single_invariant.moments.shape[1],
                                                   hidden_channels=hidden_channels,
                                                   number_of_classes=len(
                                                       {x for x in pdb_file_to_class_mapping.values()}
                                                   ),
                                                   lr=learning_rate,
                                                   epochs=epochs)

        # 6. create umap embedding based on model output
        pytorch_embedding = cls.get_embedding([train_data, test_data], model)
        umap_transformer: umap.UMAP = umap.UMAP(n_components=embedding_size)
        umap_transformer.fit(pytorch_embedding)

        # 7. store metadata

        full_output_path = Path(file_output_path).resolve()
        torch.save(model, full_output_path / "model.pth")
        pickle.dump(umap_transformer, open(str(full_output_path / "umap.pkl"), "wb"))
        meta: EmbedderMeta = EmbedderMeta(
            model_path=str(full_output_path / "model.pth"),
            umap_transformer_path=str(full_output_path / "umap.pkl"),
            pdb_folder=str(Path(pdb_file_path).resolve()),
            self_path=str(full_output_path / "meta.pkl"),
            classes_to_ids=None,  # TODO: add this later..
            invariant_types=invariant_types,
            train_acc=train_acc,
            test_acc=test_acc,
            original_invariants_file=str(full_output_path / "invariants.pkl")
        )
        pickle.dump(meta, open(str(full_output_path / "meta.pkl"), "wb"))
        pickle.dump(invariant_all, open(str(full_output_path / "invariants.pkl"), "wb"))

        return cls(list(invariant_all.keys()),
                   meta, model, pdb_target_folder=None,
                   pdb_training_folder=pdb_file_path,
                   umap_transformer=umap_transformer)

    @classmethod
    def from_model_meta_file(cls, filename: str) -> "GeometricusGraphEmbedder":
        # 1. Load model
        meta: EmbedderMeta = pickle.load(open(filename, "rb"))
        model = torch.load(meta.model_path)
        model.eval()

        # 2. Load umap
        umap_transformer = pickle.load(open(meta.umap_transformer_path))

        # 3. Original ids as further meta data to load..
        ids = [x for x in glob(str(Path(meta.pdb_folder) / "*"))]

        return cls(ids=ids, model_meta=meta, model=model,
                   pdb_training_folder=meta.pdb_folder, umap_transformer=umap_transformer)

    def pdbs_to_embedding(self, pdb_folder: str) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        invariants = self.get_multi_invariants(pdb_folder, self.model_meta.invariant_types)
        data_part1, data_part2 = transform_geometricus_dataset_for_training({x: x for x in invariants},
                                                                            invariants)
        res, _, predicted_labels, pdb_ids = self.get_embedding([data_part1, data_part2], self.model)
        return self.umap_transformer.transform(res), pdb_ids, predicted_labels

    @property
    def get_original_moment_invariants(self):
        return pickle.load(open(self.model_meta.original_invariants_file, "rb"))


if __name__ == "__main__":
    test()
