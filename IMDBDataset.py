from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, dataset, max_length=512):
        self.dataset = dataset  # Cохраняем данные

    def __getitem__(self, idx):
        return {key: val for key, val in self.dataset[idx].items()}  # Выдаём 1 элемент

    def __len__(self):
        return len(self.dataset)  # Возвращаем длину датасета
