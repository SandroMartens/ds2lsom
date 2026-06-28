from typing import Self

import networkx as nx
import numpy as np
from dbgsom.SomVQ import SomVQ
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted, validate_data


class DS2LSOM(ClusterMixin, BaseEstimator):
    """Clustering learned from vector prototypes.

    Implementation of the paper
    "Enriched topological learning for cluster detection and visualization"
    by Guénaël Cabanes, Younès Bennani, Dominique Fresneau
    10.1016/j.neunet.2012.02.019

    Parameters
    ----------
    n_prototypes: int (optional) default = inferred from data
        Number of prototypes.

    model_args : dict of dicts (optional)
        Args passed to the vector quantization algorithm.

        "init" goes to initialization.

        "train" goes to fitting/training.

    method : {"som", "kmeans"}
        Method to compute prototypes.

    threshold : int (optional), default = 1
        Number of common samples for two prototypes to be considered connected.

        Higher: More clusters.

    sigma : num (optional), default = inferred from training data
        Bandwidth parameter for local density estimation.

        Too high: All samples influence all prototypes.

        Too low: Distant samples will not influence prototypes.

    verbose : bool, default = False
        Print information about each step.

    random_state : int, RandomState instance or None, default = None
        Seed for the random number generator passed to the quantizer.
        Use an int for reproducible results.
    """

    def __init__(
        self,
        n_prototypes: int | None = None,
        threshold: int = 1,
        sigma: float | None = None,
        method: str = "som",
        verbose: bool = False,
        model_args: dict | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_prototypes = n_prototypes
        self.model_args = model_args
        self.threshold = threshold
        self.sigma = sigma
        self.verbose = verbose
        self.method = method
        self.random_state = random_state

    def fit(self, X, y=None) -> Self:
        """Fit and train SOM, enrich prototypes and return graph of prototypes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : Fitted estimator
        """
        methods = ("som", "kmeans")
        if self.method not in methods:
            raise ValueError(
                f"{self.method} is not a valid method for prototype computation."
            )

        X = validate_data(self, X)

        if self.verbose:
            print("Started training.")

        n_prototypes_ = self.n_prototypes
        if n_prototypes_ is None:
            n_prototypes_ = int(10 * (len(X) ** 0.5))
        self.n_prototypes_ = n_prototypes_

        self.quantizer_ = self._train_quantizer(X)
        self._get_dist_matrix(X)
        self.nbr_values_ = self._enrich_prototypes()
        self.edge_list_ = self._get_edges()
        self.graph_ = self._create_graph()
        self._initial_clustering()
        self._final_clustering()

        self.labels_ = self.predict(X)

        if self.verbose:
            print("Training finished.")

        return self

    def predict(self, X) -> np.ndarray:
        """Return the cluster id for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. -1 for unassigned points.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.method == "som":
            pred = pairwise_distances(self.weights_, X).argmin(axis=0)
        elif self.method == "kmeans":
            pred = self.quantizer_.predict(X)

        pred = np.array(pred, dtype=np.intp)
        for sample, prototype in enumerate(pred):
            if prototype in self.graph_:
                pred[sample] = self.graph_.nodes[prototype]["label"]
            else:
                pred[sample] = -1

        # Remap sparse prototype indices to sequential 0..n_clusters-1
        assigned = pred >= 0
        if assigned.any():
            _, pred[assigned] = np.unique(pred[assigned], return_inverse=True)
        return pred

    def _get_dist_matrix(self, X) -> None:
        """Calculate distance matrix (i, j) for prototype i and sample j."""
        if self.method == "som":
            self.dist_matrix_ = pairwise_distances(self.weights_, X)
        elif self.method == "kmeans":
            self.dist_matrix_ = self.quantizer_.transform(X).T

    def _train_quantizer(self, X) -> SomVQ | KMeans:
        """Train the vector quantizer and store weights in self.weights_."""
        if self.method == "som":
            init_kwargs: dict = {
                "max_neurons": self.n_prototypes_,
                "random_state": self.random_state,
            }
            fit_kwargs: dict = {}
            if self.model_args is not None:
                init_kwargs.update(self.model_args.get("init", {}))
                fit_kwargs.update(self.model_args.get("train", {}))
            som = SomVQ(**init_kwargs)  # type: ignore[call-overload]
            som.fit(X, **fit_kwargs)
            self.weights_ = som.weights_
            self.n_prototypes_ = len(self.weights_)
            return som

        kmeans_args_default = {
            "init": {
                "n_clusters": self.n_prototypes_,
                "random_state": self.random_state,
            },
            "train": {"sample_weight": None},
        }
        if self.model_args is not None:
            kmeans_args_default["init"].update(self.model_args.get("init", {}))
            kmeans_args_default["train"].update(self.model_args.get("train", {}))
        kmeans = KMeans(**kmeans_args_default["init"], verbose=self.verbose)
        kmeans.fit(X=X, **kmeans_args_default["train"])
        self.weights_ = kmeans.cluster_centers_
        self.n_prototypes_ = len(self.weights_)
        return kmeans

    def _enrich_prototypes(self) -> np.ndarray:
        """Enrich each prototype with density, variability, and neighborhood values."""
        nearest_prototype_id = self.dist_matrix_.argmin(axis=0)
        self.densities_ = self._estimate_density(nearest_prototype_id)
        self.variabilities_ = self._estimate_local_variability(nearest_prototype_id)
        return self._estimate_neighborhood_values()

    def _estimate_density(self, nearest_prototype_id: np.ndarray) -> np.ndarray:
        """Estimate local density for each prototype from its assigned samples."""
        if self.sigma is None:
            sigma = self._calculate_sigma()
        else:
            sigma = self.sigma

        densities = np.zeros(self.n_prototypes_)
        for prototype in range(len(self.dist_matrix_)):
            distances = self.dist_matrix_[prototype, nearest_prototype_id == prototype]
            if len(distances) > 0:
                d = np.mean(
                    (np.exp(-(distances**2) / (2 * sigma**2)))
                    / (sigma * np.sqrt(2 * np.pi))
                )
                densities[prototype] = d
            else:
                densities[prototype] = 0

        return densities

    def _calculate_sigma(self):
        """Heuristic for sigma: mean distance between prototype and nearest neighbor."""
        pairwise_prototype_distances = pairwise_distances(self.weights_, self.weights_)
        pairwise_prototype_distances.sort(axis=1)
        closest_neighbor_distances = pairwise_prototype_distances[:, 1]
        return np.mean(closest_neighbor_distances)

    def _estimate_local_variability(self, nearest_prototype_id: np.ndarray) -> np.ndarray:
        """For each prototype w, variability s is the mean distance to its assigned samples."""
        variabilities = np.zeros(self.n_prototypes_)
        for prototype in range(len(self.dist_matrix_)):
            neighbors = self.dist_matrix_[prototype, nearest_prototype_id == prototype]
            if len(neighbors) > 0:
                variabilities[prototype] = np.mean(neighbors)
            else:
                variabilities[prototype] = 0

        return variabilities

    def _estimate_neighborhood_values(self) -> np.ndarray:
        """Compute v_{i,j}: number of samples having i and j as their two closest prototypes."""
        BMUs = np.argpartition(self.dist_matrix_, 2, axis=0)[:2, :]
        v = np.zeros(shape=(self.n_prototypes_, self.n_prototypes_))
        u, counts = np.unique(BMUs, axis=1, return_counts=True)
        u = u.T
        for index, combination in enumerate(u):
            v[combination[0], combination[1]] = counts[index]
            v[combination[1], combination[0]] = counts[index]

        return v

    def _get_edges(self) -> set[tuple[int, int]]:
        """Find all prototype pairs (i, j) with v_{i,j} >= threshold."""
        indices = np.asarray(self.nbr_values_ >= self.threshold).nonzero()
        return set(zip(indices[0], indices[1]))

    def _create_graph(self) -> nx.Graph:
        """Create a graph with enriched prototype attributes."""
        g = nx.Graph()
        g.add_edges_from(self.edge_list_)

        nx.set_node_attributes(g, dict(enumerate(self.densities_)), "density")
        nx.set_node_attributes(g, dict(enumerate(self.variabilities_)), "variability")

        return g

    def _initial_clustering(self) -> None:
        """Label each prototype by gradient ascent toward the local density maximum."""
        for node in self.graph_:
            self.graph_.nodes[node]["label"] = node

        components = nx.connected_components(self.graph_)
        for component in components:
            subgraph = self.graph_.subgraph(nodes=component)
            longest_path = nx.diameter(subgraph)

            for i in range(longest_path):
                for node, nbr_data in subgraph.adj.items():
                    node_density = subgraph.nodes[node]["density"]
                    largest_gradient = 0
                    largest_gradient_neighbor = node
                    for nbr in nbr_data:
                        nbr_density = subgraph.nodes[nbr]["density"]
                        gradient = nbr_density - node_density
                        if gradient > largest_gradient:
                            largest_gradient = gradient
                            largest_gradient_neighbor = nbr

                    self.graph_.nodes[node]["label"] = self.graph_.nodes[
                        largest_gradient_neighbor
                    ]["label"]

                    subgraph.nodes[node]["label"] = self.graph_.nodes[
                        largest_gradient_neighbor
                    ]["label"]

    def _final_clustering(self) -> None:
        """Merge micro-clusters according to pairwise density threshold."""
        converged = False
        while not converged:
            G = self.graph_
            converged = True
            for e in G.edges:
                node_i = G.nodes[e[0]]
                node_j = G.nodes[e[1]]

                label_i = node_i["label"]
                label_j = node_j["label"]

                density_i = node_i["density"]
                density_j = node_j["density"]

                density_max_i = G.nodes[label_i]["density"]
                density_max_j = G.nodes[label_j]["density"]

                if density_max_i > 0 and density_max_j > 0:
                    threshold = (density_max_i**-1 + density_max_j**-1) ** -1
                else:
                    threshold = 0

                if (
                    density_i > threshold
                    and density_j > threshold
                    and label_i != label_j
                ):
                    converged = False
                    self._merge_micro_clusters(
                        G, label_i, label_j, density_max_i, density_max_j
                    )
        self.graph_ = G

    def _merge_micro_clusters(
        self,
        G: nx.Graph,
        label_i: int,
        label_j: int,
        density_max_i: float,
        density_max_j: float,
    ) -> None:
        """Overwrite label of low-density cluster with label of high-density cluster."""
        if density_max_i > density_max_j:
            new_label = label_i
            old_label = label_j
        else:
            new_label = label_j
            old_label = label_i

        for node, label in G.nodes.data("label"):
            if label == old_label:
                G.nodes[node]["label"] = new_label
