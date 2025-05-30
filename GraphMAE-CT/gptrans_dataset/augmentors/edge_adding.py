from gptrans_dataset.augmentors.augmentor import Graph, Augmentor
from gptrans_dataset.augmentors.functional import add_edge


class EdgeAdding(Augmentor):
    def __init__(self, pe: float):
        super(EdgeAdding, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        # print(edge_index.size()[1])
        edge_index = add_edge(edge_index, ratio=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
