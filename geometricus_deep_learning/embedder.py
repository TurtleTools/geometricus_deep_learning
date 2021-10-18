import typing as ty
import numpy as np
from dataclasses import dataclass
from geometricus import MomentType, MomentInvariants, SplitType
from torch_geometric.data import DataLoader
from glob import glob
import torch
from geometricus_deep_learning import dataset_utils, utils
from pathlib import Path
import pickle
from torch.utils import data
from main import train as train_model


@dataclass
class EmbedderMeta:  # TODO: save original test and train embeddings somewhere
    model_path: str
    pdb_folder: str
    self_path: str
    id_to_classname_path: str
    invariant_types: ty.List[utils.InvariantType]
    original_invariants_file: str
    dataset_path: str


@dataclass
class GeometricusGraphEmbedder:
    ids: ty.Union[ty.List[str], None]
    model_meta: ty.Union[EmbedderMeta, None] = None
    model: ty.Union[GCN, None] = None
    pdb_target_folder: ty.Union[str, None] = None
    pdb_training_folder: ty.Union[str, None] = None
    id_to_classname: ty.Union[ty.Dict[str, int]] = None
    train_set: ty.Union[DataLoader, None] = None

    @staticmethod
    def get_multi_invariants(pdb_file_path: str,
                             invariant_types: ty.List[utils.InvariantType],
                             ids_to_use: ty.Union[ty.Set[str], None] = None) -> ty.Dict[
        str, utils.MomentInvariantsSavable]:
        invariant_collection: ty.List[ty.Dict[str, utils.MomentInvariantsSavable]] = list()
        for invariant_type in invariant_types:
            invariant_collection.append(
                {k: utils.MomentInvariantsSavable.from_invariant(v) for k, v in
                 utils.invariants_from_pdb_folder(pdb_file_path, invariant_type, ids_to_use=ids_to_use).items()}
            )
        # 2. concat. different types together into single array of moments
        invariant_all: ty.Dict[str, utils.MomentInvariantsSavable] = {
            k: utils.MomentInvariantsSavable(None, None) for k in
            invariant_collection[0].keys()}

        for invariant in invariant_collection:
            invariant_all = {k: utils.concat_invariants(invariant_all[k], invariant[k]) for k in
                             invariant_all.keys()}
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
    def fit(cls,
            pdb_file_path: str,
            invariant_types: ty.List[utils.InvariantType],
            pdb_file_to_class_mapping: ty.Dict[str, str],
            hidden_channels: int = 128,
            learning_rate: float = 0.001,
            file_output_path: str = "./data/models/",
            epochs: int = 1_000,
            number_of_batches: int = 512,
            graph_distance_threshold: float = 6.) -> "GeometricusGraphEmbedder":

        # 1. create invariants according to given invariant types
        # 2. concat. different types together into single array of moments
        invariant_all = cls.get_multi_invariants(pdb_file_path, invariant_types,
                                                 ids_to_use=set(list(pdb_file_to_class_mapping.keys())))
        single_invariant = invariant_all[list(invariant_all.keys())[0]]
        # 3. Remove mappings which are not included
        pdb_file_to_class_mapping = {k: v for k, v in pdb_file_to_class_mapping.items() if k in invariant_all}

        # 4. Split into train and test sets
        train_data, class_map = utils.transform_geometricus_dataset_for_training(
            pdb_file_to_class_mapping,
            invariant_all,
            batch_no=number_of_batches,
            graph_distance_threshold=graph_distance_threshold)

        # 5. train model and store relevant metadata
        model, pytorch_embedding, labels, pdb_ids = train_model(train_data,
                                                         dataset_num_features=single_invariant.moments.shape[1],
                                                         hidden_dim=hidden_channels,
                                                         lr=learning_rate,
                                                         epochs=epochs)

        # 6. store metadata

        full_output_path = Path(file_output_path)
        torch.save(model, full_output_path / "model.pth")
        meta: EmbedderMeta = EmbedderMeta(
            model_path=str(full_output_path / "model.pth"),
            pdb_folder=str(Path(pdb_file_path)),
            self_path=str(full_output_path / "meta.pkl"),
            id_to_classname_path=str(full_output_path / "class_map.pkl"),
            invariant_types=invariant_types,
            original_invariants_file=str(full_output_path / "invariants.pkl"),
            dataset_path=str(full_output_path / "train_set.pkl")
        )
        pickle.dump(class_map, open(str(full_output_path / "class_map.pkl"), "wb"))
        pickle.dump(meta, open(str(full_output_path / "meta.pkl"), "wb"))
        pickle.dump(invariant_all, open(str(full_output_path / "invariants.pkl"), "wb"))
        pickle.dump(train_data, open(str(full_output_path / "train_set.pkl"), "wb"))

        return cls(list(pdb_ids),
                   meta, model, pdb_target_folder=None,
                   pdb_training_folder=pdb_file_path,
                   id_to_classname=class_map,
                   train_set=train_data)

    @classmethod
    def from_model_meta_file(cls, filename: str) -> "GeometricusGraphEmbedder":
        # 1. Load model
        meta: EmbedderMeta = pickle.load(open(filename, "rb"))
        model = torch.load(meta.model_path)
        model.eval()

        # 2. Original ids as further meta data to load..
        ids = [x for x in glob(str(Path(meta.pdb_folder) / "*"))]

        # 3. Load class map to retrieve real class names from ids
        class_map = pickle.load(open(meta.id_to_classname_path, "rb"))

        # 4. Load original train/test datasets

        train_data = pickle.load(open(meta.dataset_path, "rb"))

        return cls(ids=ids, model_meta=meta, model=model,
                   pdb_training_folder=meta.pdb_folder,
                   id_to_classname=class_map, train_set=train_data)

    def transform(self, pdb_folder: str,
                  mappings: ty.Union[ty.Dict[str, str], None] = None) -> ty.Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        invariants = self.get_multi_invariants(pdb_folder, self.model_meta.invariant_types)
        if mappings is None:
            dataloader, _ = utils.transform_geometricus_dataset_for_training(
                {x: x for x in invariants},
                invariants)
        else:
            dataloader, _ = utils.transform_geometricus_dataset_for_training(mappings, invariants)
        res, labels, pdb_ids = self.model.encoder.get_embeddings(dataloader)
        return res, pdb_ids, labels

    @property
    def get_original_moment_invariants(self):
        return pickle.load(open(self.model_meta.original_invariants_file, "rb"))

    def retrain(self):
        pass

    def continue_training(self):
        pass

    def get_self_embedding(self):
        return self.model.encoder.get_embeddings(self.train_set)


if __name__ == "__main__":
    pass
