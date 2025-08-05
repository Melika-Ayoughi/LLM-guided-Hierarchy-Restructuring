from itertools import product
import os

from mpmath import mp

import networkx as nx

import pandas as pd

from config import (
    TAUS,
    EMBEDDING_DIMS,
    CURVATURES,
    NCS,
    DTYPE,
)

from tree_embeddings.embeddings.constructive_method import constructively_embed_tree
from tree_embeddings.trees.file_utils import load_hierarchy
from hierarchy_utils import Hierarchy

if __name__ == "__main__":
    gen_types = ["hadamard", "optim"]
    # gen_type = "hadamard"  # optim or hadamard
    tree_kwargs = {
        # "n": 1024,
        # "r": 5,
        # "gamma": 3,
        # "seed": 30,
        # "m": 1,
    }
    curvature = 1.0
    terms = 1

    hierarchy_names = [
        # "cub_three_level",
        "ImageNet-1k",
        "pizza",
        "Core50_unbalanced",
        "Core50",
        "Madverse",
        "Matador",
        "moments_in_time",
        "BioTrove-LifeStages",
        "marine-tree",
        "NABirds",
        "COCO10k",
        "EgoObjects",
        "OpenLoris",
        "PascalVOC",
        # "Imagenet21k_v1",
        # "Visual_Genome"
        # # # "BioTrove-Balanced",
        # # # "BioTrove-Unseen",
        # # # "rare_species",
        # # # "Caltech101",
    ]
    dims = [
        # 199, 32, 64, 128, # "cub_three_level"
        40, #70, # "ImageNet-1k",
        70, # "pizza",
        70, #"Core50_unbalanced",
        20, #chatgpt:10, deepseek: 20, #"Core50",
        130, #"Madverse",
        40, #"Matador",
        70, #"moments_in_time",
        10, #"BioTrove-LifeStages",
        40, #"marine-tree",
        70, #"NABirds",
        20, #"COCO10k",
        # 40, #COCO10k ablation
        40, #"EgoObjects",
        40, #chatgpt:20 deepseek:40, #"OpenLoris",
        20, #"PascalVOC",
        # 520, # "Imagenet21k_v1",
        # 130 # "Visual_Genome"
    ]
    # Melika:
    for dim, dataset in zip(dims, hierarchy_names):
        for gen_type in gen_types:
            # for optimized in ["", "_optimized"]:
            # for optimized in ["_r1", "_r2", "_r3", "_r1_r3_r4", "_r2_r3_r4"]:
            for optimized in ["_deepseek"]:
                hierarchy_name = f"{dataset}{optimized}"
    # Mina:
    # for dataset in hierarchy_names:
    #     for dim in dims:
    #         hierarchy_name = f"{dataset}"
                graph_name = hierarchy_name

                # Load hierarchy and turn to directed if necessary
                hierarchy = load_hierarchy(
                    dataset=dataset, hierarchy_name=hierarchy_name, tree_kwargs=tree_kwargs
                )
                # todo: calculate root after node remapping
                root = [node for node in hierarchy.nodes() if hierarchy.in_degree(node) == 0][0]

                path_length = nx.dag_longest_path_length(nx.bfs_tree(hierarchy, root))
                # path_length = nx.diameter(hierarchy.to_undirected(), weight="weight")
                mp.prec = 53 * terms - terms + 1
                eps = mp.mpf(mp.eps)
                mp.prec = 2 * mp.prec
                r = 1 - eps / 2
                m = mp.log((1 + r) / (1 - r))
                tau = float(m / (1.3 * path_length))
                print("Using tau:", tau)

                if not nx.is_directed(hierarchy):
                    # Store edge weights, turn into directed graph and reassign weights
                    edge_data = {
                        (source, target): data
                        for source, target, data in nx.DiGraph(hierarchy).edges(data=True)
                    }
                    hierarchy = nx.bfs_tree(hierarchy, root)
                    nx.set_edge_attributes(hierarchy, edge_data)

                # Load original graph for evaluation purposes
                graph = load_hierarchy(dataset=dataset, hierarchy_name=graph_name, tree_kwargs=tree_kwargs)

                embeddings, rel_dist_mean, rel_dist_max = constructively_embed_tree(
                    hierarchy=hierarchy,
                    dataset=dataset,
                    hierarchy_name=hierarchy_name,
                    graph=graph,
                    embedding_dim=dim,
                    tau=tau,
                    nc=terms,
                    curvature=curvature,
                    root=root,
                    gen_type=gen_type,
                    dtype=DTYPE,
                )

                res_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "tree_embeddings",
                    "results",
                    "constructive_method",
                    dataset,
                    hierarchy_name,
                )

                print(f"Result above are for hierarchy: {hierarchy_name} using method {gen_type}")
