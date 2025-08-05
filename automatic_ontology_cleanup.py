from hierarchy_utils import Hierarchy
from evaluate import load_embedding, evaluate_tree
import json
import networkx as nx
import sys
import argparse


def filter_nodes_json(nodes_list, G):
    valid_ids = set(G.nodes)
    filtered_nodes = [node for node in nodes_list if node["id"] in valid_ids]
    return {"nodes": filtered_nodes}


def get_links_json(G):
    # Map node labels to indices if needed
    node_to_index = {node: node for node in G.nodes}

    links = []
    for source, target in G.edges:
        links.append({
            "source": node_to_index[source],
            "target": node_to_index[target]
        })
    return {"links": links}


def read_from_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def save_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


def parse_tree_text(text):
    G = nx.DiGraph()
    stack = []  # Stack to hold (depth, node)

    lines = text.strip().splitlines()
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        depth = indent // 2
        # node = int(stripped[2:])  # remove '- '
        node = stripped[2:]

        G.add_node(node)

        # If there's a parent at depth-1, add an edge
        if depth > 0:
            parent = stack[depth - 1][1]
            G.add_edge(parent, node)

        # Ensure stack is only up to this depth
        if len(stack) > depth:
            stack = stack[:depth]
        stack.append((depth, node))

    return G


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

##############################################################################
# IMAGENET:
# h = load_json("./hierarchies/ImageNet-1k/imagenet.json")
# a = read_from_file("./hierarchies/ImageNet-1k/optimized_hierarchy.txt")
# g = parse_tree_text(a)
# print(f"Depth: {max(nx.single_source_shortest_path_length(g, 1777).values())}")
# print(f"number of leaves: {len([n for n in g.nodes if g.out_degree(n) == 0])}")
# links = get_links_json(g)
# h["links"] = links["links"]
# save_json(h, "./hierarchies/ImageNet-1k/imagenet1k_optimized.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hyperbolic Embeddings')
    parser.add_argument('-dataset', type=str, default='Imagenet21k', required=True,)
    parser.add_argument('-embedding_method', type=str, default='hadamard', required=True,)
    parser.add_argument('-optimized', type=str, default="_v1_deepseek_corrected", required=True,)
    opt = parser.parse_args()

    hierarchy_names = [
                       # "ImageNet-1k",
                       # "pizza",
                       # "Core50_unbalanced",
                       # "Core50",
                       # "Madverse",
                       # "Matador",
                       # "moments_in_time",
                       # # "BioTrove-Balanced",
                       # "BioTrove-LifeStages",
                       # # "BioTrove-Unseen",
                       # "marine-tree",
                       # "NABirds",
                       # # "rare_species",
                       # # "Caltech101",
                       # "COCO10k",
                       # "EgoObjects",
                       "Imagenet21k_v1",
                       # # "Imagenet21k_v2",
                       # "imagenetood_v1",
                       # # "imagenetood_v2",
                       # "OpenLoris",
                       # # "openimages",
                       # "PascalVOC",
                       "Visual_Genome"
                       # "ot_601"
                       ]
    # embedding_methods = [
    #                     # "poincare",
    #                     # "entailment",
    #                     "optim",
    #                     # "hadamard",
    #                     ]

    # dataset = "Imagenet21k_v1"
    # hierarchy_name = f"{dataset}_deepseek"
    # adr = f"./hierarchies/{dataset}/{dataset}"
    # hierarchy = Hierarchy(f"{adr}.json", f"{dataset}")
    # hierarchy.print_hierarchy_info()
    # hierarchy.prune_hierarchy_graph_deepseek(max_depth=15)
    # hierarchy.print_hierarchy_info()
    # a = hierarchy.visualize_tree_txt(hierarchy.get_root())
    # save_to_file(a, f"./hierarchies/{dataset}/{dataset}_optimized/{hierarchy_name}/{hierarchy_name}_new.txt")

#todo: ablation deepseek:
    for dataset in hierarchy_names:
        hierarchy_name = f"{dataset}_deepseek"
        adr = f"./hierarchies/{dataset}/{dataset}"
        adr_optimized = f"./hierarchies/{dataset}/{dataset}_optimized/{hierarchy_name}/{hierarchy_name}"
        hierarchy = Hierarchy(f"{adr}.json", f"{dataset}")
        original_hierarchy_json = hierarchy.get_hierarchy_file()
        a = read_from_file(f"{adr_optimized}.txt")

        g = parse_tree_text(a)
        if nx.is_tree(g) is False:
            print(f"Graph is not a tree!")
            sys.exit(1)
        root = [node for node in g.nodes() if g.in_degree(node) == 0][0]
        print(
            f"Original depth:{hierarchy.get_depth()} current: {max(nx.single_source_shortest_path_length(g, root).values())}")
        print(
            f"Original #leaf nodes: {hierarchy.get_number_of_leaves()} current: {len([n for n in g.nodes if g.out_degree(n) == 0])}")
        print(f"Original #nodes: {hierarchy.get_number_of_nodes()}, current: {len(g.nodes)}")
        print(f"Original max degree: {hierarchy.get_degree_measures()[2]}, current: {max(dict(g.degree).values())}")
        hierarchy.print_hierarchy_info()
        links = get_links_json(g)
        nodes = filter_nodes_json(original_hierarchy_json["nodes"], g)
        original_hierarchy_json["links"] = links["links"]
        original_hierarchy_json["nodes"] = nodes["nodes"]
        save_json(original_hierarchy_json, f"{adr_optimized}.json")
        # Check the new optimized hierarchy:
        new_hierarchy = Hierarchy(f"{adr_optimized}.json", f"{hierarchy_name}_optimized")
        new_hierarchy.print_hierarchy_info()
        # new_hierarchy.visualize_tree(f'{adr_optimized}.pdf')
        # new_hierarchy.visualize_degree_histogram(f'{adr_optimized}_hist.pdf')
#todo: ablation prompt:

    # for dataset in hierarchy_names:
    #     # for optimized in ["", "_optimized"]:
    #     # for optimized in ["_optimized"]:
    #     for ablation in ["_r1", "_r2", "_r3", "_r1_r3_r4", "_r2_r3_r4"]:
    #         hierarchy_name = f"{dataset}{ablation}"
    #         adr = f"./hierarchies/{dataset}/{dataset}"
    #         adr_optimized = f"./hierarchies/{dataset}/{dataset}_optimized/{hierarchy_name}/{hierarchy_name}"
    #         hierarchy = Hierarchy(f"{adr}.json", f"{dataset}")
    #         original_hierarchy_json = hierarchy.get_hierarchy_file()
    #         a = read_from_file(f"{adr_optimized}.txt")
    #
    #         g = parse_tree_text(a)
    #         if nx.is_tree(g) is False:
    #             print(f"Graph is not a tree!")
    #             sys.exit(1)
    #         root = [node for node in g.nodes() if g.in_degree(node) == 0][0]
    #         print(
    #             f"Original depth:{hierarchy.get_depth()} current: {max(nx.single_source_shortest_path_length(g, root).values())}")
    #         print(
    #             f"Original #leaf nodes: {hierarchy.get_number_of_leaves()} current: {len([n for n in g.nodes if g.out_degree(n) == 0])}")
    #         print(f"Original #nodes: {hierarchy.get_number_of_nodes()}, current: {len(g.nodes)}")
    #         print(f"Original max degree: {hierarchy.get_degree_measures()[2]}, current: {max(dict(g.degree).values())}")
    #         hierarchy.print_hierarchy_info()
    #         links = get_links_json(g)
    #         nodes = filter_nodes_json(original_hierarchy_json["nodes"], g)
    #         original_hierarchy_json["links"] = links["links"]
    #         original_hierarchy_json["nodes"] = nodes["nodes"]
    #         save_json(original_hierarchy_json, f"{adr_optimized}.json")
    #         # Check the new optimized hierarchy:
    #         new_hierarchy = Hierarchy(f"{adr_optimized}.json", f"{hierarchy_name}_optimized")
    #         new_hierarchy.print_hierarchy_info()
    #         new_hierarchy.visualize_tree(f'{adr_optimized}.pdf')
    #         new_hierarchy.visualize_degree_histogram(f'{adr_optimized}_hist.pdf')


# # #TODO: step 1
# for hierarchy_name in hierarchy_names:
#     adr = f"./hierarchies/{hierarchy_name}/{hierarchy_name}"
#     hierarchy = Hierarchy(f"{adr}.json", f"{hierarchy_name}")
#     hierarchy.print_hierarchy_info()
#     hierarchy.visualize_tree(f'{adr}.pdf')
#     hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
#     a = hierarchy.visualize_tree_txt(hierarchy.get_root())
#     save_to_file(a, f"{adr}.txt")


# hierarchy_name = "Imagenet21k_v1"
# hierarchy_name = "Visual_Genome"
# adr = f"./hierarchies/{hierarchy_name}/{hierarchy_name}"
# hierarchy = Hierarchy(f"{adr}.json", f"{hierarchy_name}")
# hierarchy.print_hierarchy_info()
# hierarchy.prune_hierarchy_graph(max_depth=14)
# hierarchy.print_hierarchy_info()
# a = hierarchy.visualize_tree_txt(hierarchy.get_root())
# save_to_file(a, f"./hierarchies/{hierarchy_name}_optimized/{hierarchy_name}_optimized.txt")
#TODO: step 2: chatgpt

#TODO: step 3: chatgpt text to json

# for hierarchy_name in hierarchy_names:
#     adr = f"./hierarchies/{hierarchy_name}/{hierarchy_name}"
#     adr_optimized = f"./hierarchies/{hierarchy_name}/{hierarchy_name}_optimized/{hierarchy_name}_optimized"
#     hierarchy = Hierarchy(f"{adr}.json", f"{hierarchy_name}")
#     original_hierarchy_json = hierarchy.get_hierarchy_file()
#     a = read_from_file(f"{adr_optimized}.txt")
#     # g is the new flattened hierarchy in text format:
#     g = parse_tree_text(a)
#     if nx.is_tree(g) is False:
#         print(f"Graph is not a tree!")
#         sys.exit(1)
#     root = [node for node in g.nodes() if g.in_degree(node) == 0][0]
#     print(f"Original depth:{hierarchy.get_depth()} current: {max(nx.single_source_shortest_path_length(g, root).values())}")
#     print(f"Original #leaf nodes: {hierarchy.get_number_of_leaves()} current: {len([n for n in g.nodes if g.out_degree(n) == 0])}")
#     print(f"Original #nodes: {hierarchy.get_number_of_nodes()}, current: {len(g.nodes)}")
#     print(f"Original max degree: {hierarchy.get_degree_measures()[2]}, current: {max(dict(g.degree).values())}")
#     hierarchy.print_hierarchy_info()
#     links = get_links_json(g)
#     nodes = filter_nodes_json(original_hierarchy_json["nodes"], g)
#     original_hierarchy_json["links"] = links["links"]
#     original_hierarchy_json["nodes"] = nodes["nodes"]
#     save_json(original_hierarchy_json, f"{adr_optimized}.json")
#     # Check the new optimized hierarchy:
#     new_hierarchy = Hierarchy(f"{adr_optimized}.json", f"{hierarchy_name}_optimized")
#     new_hierarchy.print_hierarchy_info()
#     new_hierarchy.visualize_tree(f'{adr_optimized}.pdf')
#     new_hierarchy.visualize_degree_histogram(f'{adr_optimized}_hist.pdf')
# TODO: step 4: Generate the hyperbolic embeddings run_poincare_hiervision.sh & run_poincare_hiervision_original.sh

# TODO: step 5: Evaluate the embeddings

# for dataset in hierarchy_names:
#     for embedding_method in embedding_methods:
#         for optimized in ["", "_optimized"]:
#             if optimized == "":
#                 continue
    dataset = opt.dataset
    optimized = opt.optimized
    embedding_method = opt.embedding_method
    hierarchy_name = f"{dataset}{optimized}"
    adr = f"./hierarchies/{dataset}/{dataset}_optimized/{hierarchy_name}/{hierarchy_name}"
    adr_checkpoint = f"{adr}.bin.best"
    if embedding_method in ["entailment", "optim", "hadamard"]:
        adr = f"./tree_embeddings/data/hierarchies/{dataset}/{hierarchy_name}"
        adr_checkpoint = f"./tree_embeddings/results/constructive_method/{dataset}/{hierarchy_name}/embeddings_{embedding_method}.pt"
    tree = Hierarchy(f"{adr}.json", f"{hierarchy_name}").get_hierarchy()
    print(f"Evaluating {hierarchy_name} with method {embedding_method}....")
    embeddings = load_embedding(f"{adr_checkpoint}", f"{embedding_method}")
    # evaluate_tree(tree, embeddings)
    evaluate_tree(tree, embeddings, optimized=True)
    # dataset = opt.dataset
    # optimized = opt.optimized
    # embedding_method = opt.embedding_method
    # hierarchy_name = f"{dataset}{optimized}"
    # adr = f"./hierarchies/{dataset}/{hierarchy_name}/{hierarchy_name}"
    # adr_checkpoint = f"{adr}.bin.best"
    # if embedding_method in ["entailment", "optim", "hadamard"]:
    #     adr = f"./tree_embeddings/data/hierarchies/{dataset}/{hierarchy_name}"
    #     adr_checkpoint = f"./tree_embeddings/results/constructive_method/{dataset}/{hierarchy_name}/embeddings_{embedding_method}.pt"
    # tree = Hierarchy(f"{adr}.json", f"{hierarchy_name}").get_hierarchy()
    # print(f"Evaluating {hierarchy_name} with method {embedding_method}....")
    # embeddings = load_embedding(f"{adr_checkpoint}", f"{embedding_method}")
    # # evaluate_tree(tree, embeddings)
    # evaluate_tree(tree, embeddings, optimized=True)
##############################################################################
# # Core50 unbalanced:
# hierarchy_name = "core50"
# dataset_folder = "Core50"
# hierarchy = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}.json", f"{hierarchy_name}")
# hierarchy.print_hierarchy_info()
# adr = f"./hierarchies/{dataset_folder}/{hierarchy_name}"
# hierarchy.visualize_tree(f'{adr}.pdf')
# hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
# a = hierarchy.visualize_tree_txt(hierarchy.get_root())
# save_to_file(a, f"./hierarchies/{dataset_folder}/{hierarchy_name}.txt")
# # # TODO: Give the file to Chatgpt prompt to change it to flattened hierarchy
# original_hierarchy_json = hierarchy.get_hierarchy_file()
# a = read_from_file(f"./hierarchies/{dataset_folder}/{hierarchy_name}_flattened_hierarchy.txt")
# # g is the new flattened hierarchy in text format:
# g = parse_tree_text(a)
# if nx.is_tree(g) is False:
#     print(f"Graph is not a tree!")
#     sys.exit(1)
# root = [node for node in g.nodes() if g.in_degree(node) == 0][0]
# print(f"Original depth:{hierarchy.get_depth()} current: {max(nx.single_source_shortest_path_length(g, root).values())}")
# print(f"Original #leaf nodes: {hierarchy.get_number_of_leaves()} current: {len([n for n in g.nodes if g.out_degree(n) == 0])}")
# print(f"Original #nodes: {hierarchy.get_number_of_nodes()}, current: {len(g.nodes)}")
# links = get_links_json(g)
# nodes = filter_nodes_json(original_hierarchy_json["nodes"], g)
# original_hierarchy_json["links"] = links["links"]
# original_hierarchy_json["nodes"] = nodes["nodes"]
# save_json(original_hierarchy_json, f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized.json")
# # Check the new optimized hierarchy:
# new_hierarchy = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized.json", f"{hierarchy_name}_optimized")
# new_hierarchy.print_hierarchy_info()
# adr = f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized"
# new_hierarchy.visualize_tree(f'{adr}.pdf')
# new_hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
# # TODO: Generate the hyperbolic embeddings
# # Evaluate the embeddings:
# hierarchy_name = "core50"
# dataset_folder = "Core50"
# tree = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}.json", f"{hierarchy_name}").get_hierarchy()
# embeddings = load_embedding(f"./hierarchies/{dataset_folder}/{hierarchy_name}_dim20.bin.best", "poincare")
# evaluate_tree(tree, embeddings)
#
# hierarchy_name = "core50_optimized"
# dataset_folder = "Core50"
# tree = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}.json", f"{hierarchy_name}").get_hierarchy()
# embeddings = load_embedding(f"./hierarchies/{dataset_folder}/{hierarchy_name}_dim20.bin.best", "poincare")
# evaluate_tree(tree, embeddings)
##############################################################################
# Core50 unbalanced:
# hierarchy_name = "pizza"
# dataset_folder = "pizza"
# hierarchy = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}.json", f"{hierarchy_name}")
# hierarchy.print_hierarchy_info()
# adr = f"./hierarchies/{dataset_folder}/{hierarchy_name}"
# hierarchy.visualize_tree(f'{adr}.pdf')
# hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
# a = hierarchy.visualize_tree_txt(hierarchy.get_root())
# save_to_file(a, f"./hierarchies/{dataset_folder}/{hierarchy_name}.txt")
# # TODO: Give the file to Chatgpt prompt to change it to flattened hierarchy
# original_hierarchy_json = hierarchy.get_hierarchy_file()
# a = read_from_file(f"./hierarchies/{dataset_folder}/{hierarchy_name}_flattened.txt")
# # g is the new flattened hierarchy in text format:
# g = parse_tree_text(a)
# if nx.is_tree(g) is False:
#     print(f"Graph is not a tree!")
#     sys.exit(1)
# root = [node for node in g.nodes() if g.in_degree(node) == 0][0]
# print(f"Original depth:{hierarchy.get_depth()} current: {max(nx.single_source_shortest_path_length(g, root).values())}")
# print(f"Original #leaf nodes: {hierarchy.get_number_of_leaves()} current: {len([n for n in g.nodes if g.out_degree(n) == 0])}")
# print(f"Original #nodes: {hierarchy.get_number_of_nodes()}, current: {len(g.nodes)}")
# links = get_links_json(g)
# nodes = filter_nodes_json(original_hierarchy_json["nodes"], g)
# original_hierarchy_json["links"] = links["links"]
# original_hierarchy_json["nodes"] = nodes["nodes"]
# save_json(original_hierarchy_json, f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized.json")
# # Check the new optimized hierarchy:
# new_hierarchy = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized.json", f"{hierarchy_name}_optimized")
# new_hierarchy.print_hierarchy_info()
# adr = f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized"
# new_hierarchy.visualize_tree(f'{adr}.pdf')
# new_hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
# # TODO: Generate the hyperbolic embeddings
# # Evaluate the embeddings:
# hierarchy_name = "pizza_optimized"
# dataset_folder = "pizza"
# tree = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}.json", f"{hierarchy_name}").get_hierarchy()
# embeddings = load_embedding(f"./hierarchies/{dataset_folder}/{hierarchy_name}_dim70.bin.best", "poincare")
# evaluate_tree(tree, embeddings)

##############################################################################
# Core50 unbalanced:
# hierarchy_name = "core50_unbalanced"
# dataset_folder = "Core50"
# hierarchy = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}.json", f"{hierarchy_name}")
# hierarchy.print_hierarchy_info()
# adr = f"./hierarchies/{dataset_folder}/{hierarchy_name}"
# hierarchy.visualize_tree(f'{adr}.pdf')
# hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
# a = hierarchy.visualize_tree_txt(hierarchy.get_root())
# save_to_file(a, f"./hierarchies/{dataset_folder}/{hierarchy_name}.txt")
# # TODO: Give the file to Chatgpt prompt to change it to flattened hierarchy
# original_hierarchy_json = hierarchy.get_hierarchy_file()
# a = read_from_file(f"./hierarchies/{dataset_folder}/{hierarchy_name}_flattened_hierarchy.txt")
# # g is the new flattened hierarchy in text format:
# g = parse_tree_text(a)
# if nx.is_tree(g) is False:
#     print(f"Graph is not a tree!")
#     sys.exit(1)
# root = [node for node in g.nodes() if g.in_degree(node) == 0][0]
# print(f"Original depth:{hierarchy.get_depth()} current: {max(nx.single_source_shortest_path_length(g, root).values())}")
# print(f"Original #leaf nodes: {hierarchy.get_number_of_leaves()} current: {len([n for n in g.nodes if g.out_degree(n) == 0])}")
# print(f"Original #nodes: {hierarchy.get_number_of_nodes()}, current: {len(g.nodes)}")
# links = get_links_json(g)
# nodes = filter_nodes_json(original_hierarchy_json["nodes"], g)
# original_hierarchy_json["links"] = links["links"]
# original_hierarchy_json["nodes"] = nodes["nodes"]
# save_json(original_hierarchy_json, f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized.json")
# # Check the new optimized hierarchy:
# new_hierarchy = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized.json", f"{hierarchy_name}_optimized")
# new_hierarchy.print_hierarchy_info()
# adr = f"./hierarchies/{dataset_folder}/{hierarchy_name}_optimized"
# new_hierarchy.visualize_tree(f'{adr}.pdf')
# new_hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
# # TODO: Generate the hyperbolic embeddings
# # Evaluate the embeddings:
# hierarchy_name = "core50_unbalanced"
# dataset_folder = "Core50"
# tree = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}.json", f"{hierarchy_name}").get_hierarchy()
# embeddings = load_embedding(f"./hierarchies/{dataset_folder}/{hierarchy_name}_dim70.bin.best", "poincare")
# evaluate_tree(tree, embeddings)
#
# hierarchy_name = "core50_unbalanced_optimized"
# dataset_folder = "Core50"
# tree = Hierarchy(f"./hierarchies/{dataset_folder}/{hierarchy_name}.json", f"{hierarchy_name}").get_hierarchy()
# embeddings = load_embedding(f"./hierarchies/{dataset_folder}/{hierarchy_name}_dim70.bin.best", "poincare")
# evaluate_tree(tree, embeddings)
##############################################################################

# hierarchy = Hierarchy("./hierarchies/ImageNet-1k/imagenet.json", "imagenet1k")
# # reorganized_hierarchy = Hierarchy("./hierarchies/ImageNet-1k/imagenet_reorganized.json", "imagenet1k_reorganized")
# # hierarchy = Hierarchy("./hierarchies/pizza/pizza_1lesshight.json", "pizza_1lesshight")
# # a = hierarchy.visualize_tree_txt(hierarchy.get_root())
# # hierarchy.visualize_compact_tree()
# # hierarchy.visualize_depth_histogram()
# output = hierarchy.transform_hierarchy(level=1)
# print(output)
# save_to_file(output, "./hierarchies/ImageNet-1k/imagenet-1.txt")
# a = read_from_file("./hierarchies/ImageNet-1k/imagenet-1-copy.txt")
# g = parse_tree_text(a)
#
# # save_json(output, "./hierarchies/pizza/pizza_candidates.json")
# reorganized_hierarchy = Hierarchy("./hierarchies/pizza/pizza_1.json", "pizza_1")
# # reorganized_hierarchy_2 = Hierarchy("./hierarchies/pizza/pizza_1_topdown.json", "pizza_1_topdown")
#
#
# hierarchy.print_hierarchy_info()
# reorganized_hierarchy.print_hierarchy_info()
# # reorganized_hierarchy_2.print_hierarchy_info()
#
# # Original tree
# # adr = "./hierarchies/ImageNet-1k/imagenet"
# adr = "./hierarchies/pizza/pizza"
# hierarchy.visualize_tree(f'{adr}.pdf')
# hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
#
# # Automatic cleanup
# # adr = "./hierarchies/ImageNet-1k/imagenet_reorganized"
# adr = "./hierarchies/pizza/pizza_1"
# reorganized_hierarchy.visualize_tree(f'{adr}.pdf')
# reorganized_hierarchy.visualize_degree_histogram(f'{adr}_hist.pdf')
# # Evaluate the cleaned up tree
#
# # Generate embeddings
#
# # Evaluate the original tree
# # evaluate_tree(hierarchy.get_hierarchy, embedding?)

