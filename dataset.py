from torch.utils.data import Dataset

"""
Standard Pytorch Dataset class for loading datasets.
"""


class STSDataset(Dataset):
    def __init__(
        self,
        sent1_tensor,
        sent2_tensor,
        target_tensor,
        sents1_length_tensor,
        sents2_length_tensor,
    ):
        """
        initializes  and populates the the length, data and target tensors, and raw texts list
        """
        assert (
            sent1_tensor.size(0)
            == target_tensor.size(0)
            == sent2_tensor.size(0)
            == sents1_length_tensor.size(0)
            == sents2_length_tensor.size(0)
        )
        self.sent1_tensor = sent1_tensor
        self.sent2_tensor = sent2_tensor
        self.target_tensor = target_tensor
        self.sents1_length_tensor = sents1_length_tensor
        self.sents2_length_tensor = sents2_length_tensor
        # Note: removed raw texts as would need to pad those too; batching does not work without it

    def __getitem__(self, index):
        """
        returns the tuple of data tensor, targets, lengths of sequences tensor and raw texts list
        """
        return (
            self.sent1_tensor[index],
            self.sent2_tensor[index],
            self.sents1_length_tensor[index],
            self.sents2_length_tensor[index],
            self.target_tensor[index],
        )

    def __len__(self):
        """
        returns the length of the data tensor.
        """
        return self.target_tensor.size(0)
