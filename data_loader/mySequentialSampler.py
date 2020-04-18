from torch.utils.data.sampler import SequentialSampler

# extend SequentialSampler to address the limitations of the vanilla class
class mySquentialSampler(SequentialSampler):
    def __init__(self, data_source):
        super().__init__(data_source)

    def __iter__(self):
        return iter(self.data_source)
