# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations, batch_size=1):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        - batch_size: number of episodes per batch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        self.batch_size = batch_size

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it
        bs = self.batch_size

        # Adjust iterations to account for batch size
        # If iterations=100 and bs=4, we yield 25 batches
        for it in range(self.iterations // bs):
            total_batch = []
            for _ in range(bs):
                batch_size = spc * cpi
                batch = torch.LongTensor(batch_size)
                c_idxs = torch.randperm(len(self.classes))[:cpi]
                for i, c in enumerate(self.classes[c_idxs]):
                    s = slice(i * spc, (i + 1) * spc)
                    label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                    sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                    batch[s] = self.indexes[label_idx][sample_idxs]
                # FEAT usually keeps samples of the same class together in the batch
                # but ProtoNet's loss function handles unique labels.
                # To be safe and efficient, we just collect them.
                total_batch.append(batch)
            
            yield torch.cat(total_batch)

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
