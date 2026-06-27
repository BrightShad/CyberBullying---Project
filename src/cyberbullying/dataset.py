import torch
from torch.utils.data import Dataset

from cyberbullying.preprocessing import Vocab


class TweetDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int, task: str = "binary"):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.task = task

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long)
        dtype = torch.float if self.task == "binary" else torch.long
        label = torch.tensor(self.labels[idx], dtype=dtype)
        return input_ids, label


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels
