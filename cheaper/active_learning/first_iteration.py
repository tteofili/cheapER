"""
First-iteration selection for Battleship-style active learning (cold start).
Selects a budget of candidate pairs to send to the oracle without using a trained matcher.
"""
import logging
import random
from typing import List, Tuple, Any, Optional

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def _pair_to_text(pair: Tuple[Any, Any]) -> str:
    """Turn (tuple1, tuple2) into a single string for embedding. tuple1/tuple2 are lists of attribute values."""
    t1, t2 = pair[0], pair[1]
    parts = []
    for a in t1:
        parts.append(str(a))
    for a in t2:
        parts.append(str(a))
    return " ".join(parts)


def _embed_tfidf(candidates: List[Tuple[Any, Any]], max_features: int = 5000) -> np.ndarray:
    """Embed each candidate as tf-idf vector over the concatenated attribute text."""
    texts = [_pair_to_text(p) for p in candidates]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english", token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(texts)
    return X.toarray()


def select_random(
    candidates: List[Tuple[Any, Any]],
    budget: int,
    seed: int = None,
) -> List[int]:
    """
    Select a random subset of indices of size min(budget, len(candidates)).
    Returns list of indices into candidates.
    """
    if seed is not None:
        random.seed(seed)
    n = min(budget, len(candidates))
    return random.sample(range(len(candidates)), n)


def select_diversity(
    candidates: List[Tuple[Any, Any]],
    budget: int,
    seed: int = None,
    n_clusters: int = None,
    use_sbert: bool = False,
) -> List[int]:
    """
    Select a diverse subset by embedding candidates, clustering, and sampling from each cluster.
    Uses tf-idf by default; if use_sbert=True and sentence_transformers is available, uses SBERT.
    Returns list of indices into candidates.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    n = min(budget, len(candidates))
    if n == 0:
        return []

    if use_sbert:
        try:
            from sentence_transformers import SentenceTransformer
            texts = [_pair_to_text(p) for p in candidates]
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(texts)
        except ImportError:
            logger.warning("sentence_transformers not installed; falling back to tf-idf for diversity selection")
            embeddings = _embed_tfidf(candidates)
    else:
        embeddings = _embed_tfidf(candidates)

    if n_clusters is None:
        n_clusters = min(max(2, n // 5), len(candidates), 50)
    n_clusters = min(n_clusters, len(candidates), n)
    if n_clusters < 1:
        return select_random(candidates, budget, seed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    cluster_ids = list(range(n_clusters))
    # cluster_sizes = [np.sum(labels == c) for c in cluster_ids]
    # Sample proportionally from each cluster; at least 1 from non-empty clusters if budget allows
    selected = []
    per_cluster = max(1, n // n_clusters)
    remainder = n - per_cluster * n_clusters
    for c in cluster_ids:
        indices_c = np.where(labels == c)[0].tolist()
        if not indices_c:
            continue
        take = min(per_cluster + (1 if remainder > 0 else 0), len(indices_c), n - len(selected))
        if take <= 0:
            break
        chosen = random.sample(indices_c, take)
        selected.extend(chosen)
        if remainder > 0:
            remainder -= 1
        if len(selected) >= n:
            break
    # If we got fewer than n (e.g. many tiny clusters), fill with random from rest
    selected = list(set(selected))[:n]
    if len(selected) < n:
        remaining = [i for i in range(len(candidates)) if i not in selected]
        need = n - len(selected)
        selected.extend(random.sample(remaining, min(need, len(remaining))))
    return selected[:n]


def select_centrality(
    candidates: List[Tuple[Any, Any]],
    budget: int,
    seed: int = None,
    n_clusters: int = None,
    use_sbert: bool = False,
    criterion: str = "pagerank",
    nn_param: int = 15,
    edges_ratio: float = 0.03,
) -> List[int]:
    """
    Select by graph centrality (Battleship-style): embed, cluster, build k-NN graph
    within clusters (edge weight = cosine similarity), compute centrality per connected
    component, distribute budget across CCs, take top nodes by centrality.
    Returns list of indices into candidates.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    n = min(budget, len(candidates))
    if n == 0:
        return []
    if len(candidates) <= budget:
        return list(range(len(candidates)))

    # 1. Embed and normalize
    if use_sbert:
        try:
            from sentence_transformers import SentenceTransformer
            texts = [_pair_to_text(p) for p in candidates]
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(texts)
        except ImportError:
            logger.warning("sentence_transformers not installed; falling back to tf-idf for centrality selection")
            embeddings = _embed_tfidf(candidates)
    else:
        embeddings = _embed_tfidf(candidates)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_norm = embeddings / norms

    # 2. Cluster
    if n_clusters is None:
        n_clusters = min(max(2, n // 5), len(candidates), 50)
    n_clusters = min(n_clusters, len(candidates))
    if n_clusters < 1:
        n_clusters = 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings_norm)

    # 3. Build graph: nodes = indices, edges = k-NN within each cluster, weight = cosine (inner product on normalized)
    G = nx.Graph()
    G.add_nodes_from(range(len(candidates)))
    k_neighbors = min(nn_param + 1, len(candidates))
    for c in range(n_clusters):
        indices_c = np.where(labels == c)[0]
        if len(indices_c) < 2:
            continue
        emb_c = embeddings_norm[indices_c]
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(indices_c)), metric="cosine", algorithm="brute")
        nbrs.fit(emb_c)
        dists, neighs = nbrs.kneighbors(emb_c)
        for i, idx in enumerate(indices_c):
            for j_local, d in zip(neighs[i], dists[i]):
                if j_local == i:
                    continue
                j_global = indices_c[j_local]
                if idx == j_global:
                    continue
                w = max(1.0 - float(d), 0.0)
                if w <= 0:
                    continue
                u, v = min(idx, j_global), max(idx, j_global)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=w)

    # 4. Connected components
    ccs = list(nx.connected_components(G))
    if not ccs:
        return select_random(candidates, budget, seed)
    total_nodes = sum(len(cc) for cc in ccs)
    if total_nodes == 0:
        return select_random(candidates, budget, seed)

    # 5. Centrality per CC and distribute budget
    budget_per_cc = []
    remainder = budget
    for cc in ccs:
        take = int(budget * len(cc) / total_nodes)
        budget_per_cc.append(min(take, len(cc)))
        remainder -= budget_per_cc[-1]
    idx_by_size = np.argsort([-len(cc) for cc in ccs])
    for i in range(remainder):
        if remainder <= 0:
            break
        ii = idx_by_size[i % len(ccs)]
        if budget_per_cc[ii] < len(ccs[ii]):
            budget_per_cc[ii] += 1
            remainder -= 1

    # 6. Select top by centrality in each CC
    selected = []
    for cc, b_cc in zip(ccs, budget_per_cc):
        if b_cc <= 0 or not cc:
            continue
        sub = G.subgraph(cc)
        if criterion == "bc":
            try:
                cent = nx.betweenness_centrality(sub, weight="weight", normalized=True)
            except Exception:
                cent = {n: 1.0 / len(cc) for n in sub.nodes()}
        else:
            try:
                cent = nx.pagerank(sub, weight="weight")
            except Exception:
                cent = {n: 1.0 / len(cc) for n in sub.nodes()}
        sorted_nodes = sorted(cent.keys(), key=lambda x: cent[x], reverse=True)
        take = min(b_cc, len(sorted_nodes))
        selected.extend(sorted_nodes[:take])

    selected = list(dict.fromkeys(selected))
    if len(selected) < n:
        all_nodes = set(range(len(candidates)))
        remaining = list(all_nodes - set(selected))
        need = n - len(selected)
        selected.extend(random.sample(remaining, min(need, len(remaining))))
    return selected[:n]


def first_iteration_select(
    candidates: List[Tuple[Any, Any]],
    budget: int,
    strategy: str = "random",
    seed: int = None,
    n_clusters: int = None,
    use_sbert: bool = False,
    criterion: str = "pagerank",
    nn_param: int = 15,
    edges_ratio: float = 0.03,
) -> List[int]:
    """
    First-iteration selection: choose `budget` indices from `candidates`.
    strategy: 'random', 'diversity', or 'centrality'.
    Returns list of indices into candidates (length min(budget, len(candidates))).
    """
    if not candidates:
        return []
    if strategy == "random":
        return select_random(candidates, budget, seed)
    elif strategy == "diversity":
        return select_diversity(candidates, budget, seed=seed, n_clusters=n_clusters, use_sbert=use_sbert)
    elif strategy == "centrality":
        return select_centrality(
            candidates,
            budget,
            seed=seed,
            n_clusters=n_clusters,
            use_sbert=use_sbert,
            criterion=criterion,
            nn_param=nn_param,
            edges_ratio=edges_ratio,
        )
    else:
        raise ValueError("strategy must be 'random', 'diversity', or 'centrality'")
