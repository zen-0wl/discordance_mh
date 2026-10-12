import pickle

import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class EmbeddingDataset(Dataset):

    def __init__(self, file_path):

        with open(file_path, "rb") as f:

            self.data = pickle.load(f)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        sequence = torch.tensor(

            sample["embedding_sequence"],
            dtype=torch.float32,
        )

        label = torch.tensor(

            sample["label"],
            dtype=torch.long,
        )

        return {

            "subject_id": sample["subject_id"],

            "embedding_sequence": sequence,

            "sequence_length": sample["sequence_length"],

            "label": label,
        }
        
def collate_fn(batch):

    max_len = max(

        sample["sequence_length"]

        for sample in batch
    )

    embedding_dim = batch[0]["embedding_sequence"].shape[-1]

    embeddings = torch.zeros(

        len(batch),

        max_len,

        embedding_dim,
    )

    attention_mask = torch.zeros(

        len(batch),

        max_len,
    )

    labels = torch.zeros(

        len(batch),

        dtype=torch.long,
    )

    subject_ids = []

    lengths = []

    for i, sample in enumerate(batch):

        seq = sample["embedding_sequence"]

        length = seq.shape[0]

        embeddings[i, :length] = seq

        attention_mask[i, :length] = 1

        labels[i] = sample["label"]

        lengths.append(length)

        subject_ids.append(sample["subject_id"])

    return {

        "subject_id": subject_ids,

        "embedding_sequence": embeddings,

        "attention_mask": attention_mask,

        "sequence_length": torch.tensor(lengths),

        "label": labels,
    }

def create_dataloader(

    file_path,

    batch_size=8,

    shuffle=False,

    num_workers=0,
):

    dataset = EmbeddingDataset(

        file_path
    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        collate_fn=collate_fn,

        num_workers=num_workers,
    )

    return loader