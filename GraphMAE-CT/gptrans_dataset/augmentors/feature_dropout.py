from gptrans_dataset.augmentors.augmentor import Graph, Augmentor
from gptrans_dataset.augmentors.functional import dropout_feature


class FeatureDropout(Augmentor):
    def __init__(self, pf: float):
        super(FeatureDropout, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        x = dropout_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
