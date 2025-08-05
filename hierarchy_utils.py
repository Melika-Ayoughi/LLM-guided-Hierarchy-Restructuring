import networkx as nx
import json
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict
from openai import OpenAI
from getpass import getpass


class Hierarchy:
    def __init__(self, hierarchy_file, hierarchy_name):
        self.hierarchy_file = hierarchy_file
        self.hierarchy_name = hierarchy_name
        self.graph = self.load_hierarchy_into_nx(hierarchy_file)
        if nx.is_tree(self.graph) is False:
            print(f"Graph {self.hierarchy_name} is not a tree!")
            sys.exit(1)
        self.root = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0][0]
        dir_part = os.path.dirname(hierarchy_file)
        self.save_adjacency(address=os.path.join(dir_part, f"{hierarchy_name}_adjacency.csv"))

    def load_hierarchy_file(self, json_file):
        """
        Load a hierarchy file and return the graph object.
        Args:
            json_file (str): Path to the hierarchy JSON file.
        Returns:
            json_file
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data

    def get_hierarchy_file(self):
        """
        Load a hierarchy file and return the graph object.
        Returns:
            json_file
        """
        with open(self.hierarchy_file, 'r') as f:
            data = json.load(f)
        return data

    def load_hierarchy_into_nx(self, json_file):
        """
        Load a hierarchy file and return the graph object.
        Args:
            json_file (str): Path to the hierarchy JSON file.
        Returns:
            nx.DiGraph: A directed graph representing the hierarchy.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        for node in data["nodes"]:
            G.add_node(node["id"])

        # Add edges
        for link in data["links"]:
            G.add_edge(link["source"], link["target"])

        return G

    def get_hierarchy_name(self):
        return self.hierarchy_name

    def get_hierarchy(self):
        return self.graph

    def get_root(self):
        return self.root

    def get_number_of_nodes(self):
        """
        Get the number of nodes in the hierarchy.
        Returns:
            int: Number of nodes in the hierarchy.
        """
        return self.graph.number_of_nodes()

    def get_number_of_edges(self):
        return self.graph.number_of_edges()

    def get_depth(self):
        """
        Get the depth of the hierarchy.
        Returns:
            int: Depth of the hierarchy.
        """
        lengths = nx.single_source_shortest_path_length(self.graph, self.root)
        return max(lengths.values())

    def get_number_of_leaves(self):
        """
        Get the number of leaf nodes in the hierarchy.
        Returns:
            int: Number of leaf nodes in the hierarchy.
        """
        return len([n for n in self.graph.nodes if self.graph.out_degree(n) == 0])

    def get_average_branching_factor_excluding_leaf(self):
        """
        Get the average branching factor of the hierarchy.
        Returns:
            float: Average branching factor of the hierarchy.
        Explanation:
        The average branching factor is calculated as the average number of children per internal node (meaning excluding leaf nodes).
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0
        total_children = sum(self.graph.out_degree(n) for n in self.graph.nodes if self.graph.out_degree(n) > 0)
        internal_nodes_count = sum(1 for n in self.graph.nodes if self.graph.out_degree(n) > 0)
        if internal_nodes_count == 0:
            return 0.0
        return total_children / internal_nodes_count

    def get_depth_measures(self):
        # Identify leaf nodes (nodes with degree 1, excluding the root)
        leaf_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) == 1 and n != self.root]
        # Calculate depth of each leaf
        depths = [nx.shortest_path_length(self.graph, self.root, leaf) for leaf in leaf_nodes]
        # Calculate average depth
        sakin_index = sum(depths)
        average_depth = sum(depths) / len(depths)  # normalize by number of leaf nodes
        variance_depth = sum((d - average_depth) ** 2 for d in depths) / len(depths)
        return sakin_index, average_depth, variance_depth, math.sqrt(variance_depth)

    def get_degree_measures(self):
        degree_counts = nx.degree_histogram(self.graph)
        degrees = range(len(degree_counts))
        max_degree = max(dict(self.graph.degree).values())
        average_degree = sum(dict(self.graph.degree).values()) / self.graph.number_of_nodes()
        return degrees, degree_counts, max_degree, average_degree

    def get_average_vertex_depth(self):
        depths = nx.single_source_shortest_path_length(self.graph, self.root).values()
        # Calculate average depth
        average_depth = sum(depths) / len(depths)
        return average_depth

    def get_smallest_path_from_root(self):
        shortest_path_lengths = nx.single_source_shortest_path_length(self.graph, self.root)
        # Exclude the root and find the minimum shortest path length
        min_path_length = min(length for node, length in shortest_path_lengths.items() if node != self.root)
        return min_path_length

    def save_adjacency(self, address):
        if os.path.exists(address):
            return
        edges = list(self.graph.edges())

        # Create a DataFrame and add the weight column
        df = pd.DataFrame(edges, columns=["id1", "id2"])
        df["weight"] = 1  # Add the weight column with value 1

        # Save to a CSV file without row indices
        df.to_csv(address, index=False)

    def visualize_degree_histogram(self, address):
        degrees, degree_counts, _, _ = self.get_degree_measures()
        plt.bar(degrees, degree_counts, color='#1B4C68')
        plt.title("Degree Histogram")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        # plt.ylim(0, 300)
        plt.savefig(address)
        plt.clf()

    def visualize_tree(self, address):
        plt.figure(figsize=(15, 10))
        nx.nx_agraph.write_dot(self.graph, 'test.dot')
        # nx.draw(tree, pos, with_labels=True, arrows=True, font_size=8, node_size=150, node_color="#4b9396")
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
        nx.draw(self.graph, pos, with_labels=False, node_size=150, font_size=8, node_color="#4b9396")
        # for node, (x, y) in pos.items():
        #     plt.text(
        #         x, y, s=node, #node["name"]
        #         horizontalalignment='center',
        #         verticalalignment='top',
        #         rotation=90,  # Rotate 90 degrees
        #         fontsize=8  # Smaller text
        #     )
        plt.savefig(address)
        plt.clf()

    def get_hierarchy_info(self):
        hierarchy_info = {}
        sakin_index, average_depth, variance_depth, std_depth = self.get_depth_measures()
        degrees, degree_counts, max_degree, average_degree = self.get_degree_measures()
        hierarchy_info[self.hierarchy_name] = {
            "hierarchy_name": self.hierarchy_name,
            "number_of_nodes": self.get_number_of_nodes(),
            "number_of_edges": self.get_number_of_edges(),
            "depth": self.get_depth(),
            "number_of_leaves": self.get_number_of_leaves(),
            "average_branching_factor_nonleaf": self.get_average_branching_factor_excluding_leaf(),
            "Sackin index": sakin_index,
            "Average depth": average_depth,
            "std": std_depth,
            "Variance depth": variance_depth,
            "Average vertex depth": self.get_average_vertex_depth(),
            "Max degree": max_degree,
            "Average degree": average_degree,
            "Normalized height imbalance": (self.get_depth() - self.get_smallest_path_from_root()) / self.get_depth(),
        }
        return hierarchy_info

    def print_hierarchy_info(self):
        for k, v in self.get_hierarchy_info().items():
            print(f"{k}: {v}")

    def visualize_compact_tree(self):
        """Compact text visualization of hierarchy"""
        # root = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0][0]

        levels = defaultdict(list)
        for node in nx.topological_sort(self.graph):
            if node == self.root:
                levels[0].append(node)
            else:
                pred = next(self.graph.predecessors(node))  # Get parent
                level = max(l for l, nodes in levels.items() if pred in nodes) + 1
                levels[level].append(node)

        # Build compact representation
        summary = []
        for level in sorted(levels):
            nodes = levels[level]
            summary.append(f"[Level {level}] {len(nodes)} nodes: {nodes[:3]}{'...' if len(nodes) > 10 else ''}")

        print("COMPACT HIERARCHY VISUALIZATION:")
        print("\n".join(summary))
        print(f"\nTotal depth: {max(levels.keys())}")

    def visualize_depth_histogram(self):
        """Depth distribution histogram"""

        depths = nx.shortest_path_length(self.graph, self.root)
        depth_counts = defaultdict(int)

        for node, depth in depths.items():
            depth_counts[depth] += 1

        print("DEPTH DISTRIBUTION HISTOGRAM:")
        for depth in sorted(depth_counts):
            print(f"[Depth {depth}]: {depth_counts[depth]} nodes")

    def visualize_tree_txt(self, node, prefix=""):
        lines = [f"{prefix}- {node}"]
        for child in self.graph.successors(node):
            lines.append(self.visualize_tree_txt(child, prefix + "  "))
        return "\n".join(lines)

    def transform_hierarchy(self, level):
        links = self.load_hierarchy_file(self.hierarchy_file)["links"]
        max_retries = 1
        # MODEL = "deepseek/deepseek-chat-v3-0324:free"  # Model name on OpenRouter
        MODEL = "gpt-4o-mini"
        openai_api_key = getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        client = OpenAI()

        prompt = (
            f"TASK: Make the {self.get_depth()}-deep hierarchy shallower by {level} level(s) using these recommendations:\n\n"
            f"INPUT TREE:\n"
            f"{self.visualize_tree_txt(self.root)}\n\n"
            f"RECOMMENDATIONS:\n"
            f"- Design hierarchies for width: The most effective embedding algorithms leverage the hierarchical order between nodes to generate embeddings. Consequently, these algorithms perform best with wide hierarchies that have high branching factors, rather than deep narrow trees with slower branching.\n"
            f"- Do not worry about balance: Current algorithms are largely agnostic to the balance between subtrees. Interestingly, our findings indicate that when balance is not prioritized or feasible, embedding performance is not significantly impacted. It is better to have a wide imbalanced hierarchy than a deep balanced hierarchy. Achieving both high width and balance leads to the best performance.\n"
            f"- Hyperbolic embeddings can handle additional node complexity: Do no remove or rename nodes, only move nodes.\n"
            f"- Avoid multiple inheritance: While Poincaré embeddings can handle hierarchies with multiple inheritance, high-performance embedding algorithms do not support them. Therefore, to minimize distortion, it is best to have single inheritance. This approach is also recommended in many current ontology evaluation methodologies.\n"
            f"INSTRUCTIONS:\n"
            f"1. Think step by step.\n"
            f"2. Return the modified hierarchy in the same format as input tree.\n"
            # f"1. Analyze this compressed tree representation\n"
            # f"2. Identify promotion candidates (prioritize single-child nodes)\n"
            # f"3. Return ONLY:\n"
            # f"   a) Nodes to promote (format: 'CANDIDATES: [node ids]')\n"
            # f"   b) Depth before/after (format: 'DEPTH: 5→3')\n"
            # f"   c) New parent assignments for promoted nodes only (format: 'child:new parent')\n\n"
            #
            # f"RULES:\n"
            # f"- Never modify node content\n"
            # f"- Preserve all original nodes\n"
            # f"- Start from deepest nodes first\n"
            # f"- Single-child nodes are top priority"
        )
        print(prompt)
        # prompt = (
        #     f"TASK: Transform this JSON tree with nodes and links to make it {level} level(s) shallower by promoting children nodes "
        #     f"starting from leaf nodes and moving upward. Prioritize nodes with single children.\n\n"
        #     f"INPUT TREE:\n"
        #     f"{load_json(hierarchy)}\n\n"
        #     f"REQUIREMENTS:\n"
        #     f"1. WORK BOTTOM-UP: Start from leaf nodes and move upward\n"
        #     f"2. SINGLE-CHILD PRIORITY: First merge nodes with exactly one child\n"
        #     f"3. PRESERVE CONTENT: Never modify node data, only change parent-child relationships\n"
        #     f"4. STRUCTURAL CHANGE: You MUST modify the links in the output JSON\n\n"
        #     f"TRANSFORMATION STEPS:\n"
        #     f"1. Calculate current depth and visualize structure\n"
        #     f"2. Identify all leaf nodes (no children)\n"
        #     f"3. For each leaf, examine its parent:\n"
        #     f"   a) If parent has only this child: promote child to parent's level\n"
        #     f"   b) If parent has multiple children: consider promoting if needed for depth reduction\n"
        #     f"4. Repeat until depth is reduced by exactly {level} level(s)\n"
        #     f"5. Verify all original nodes are preserved with same data\n\n"
        #     f"OUTPUT FORMAT:\n"
        #     f"```json\n"
        #     f"{{  \n"
        #     f"  \"explanation\": \"Brief description of changes made\",\n"
        #     f"  \"original_depth\": N,\n"
        #     f"  \"new_depth\": ,\n"
        #     f"  \"modified_tree\": {{links: [...]}}  // THE ACTUAL MODIFIED JSON STRUCTURE\n"
        #     f"}}\n"
        #     f"```\n\n"
        #     f"IMPORTANT:\n"
        #     f"- You MUST change the parent-child relationships in the output JSON in links\n"
        #     f"- If no changes were made, explain why and return original structure\n"
        #     f"- Never omit nodes or modify their content\n"
        #     f"- Pay special attention to arrays of children in the JSON structure"
        # )

        # prompt = (
        #     f"You are given a JSON representation of a hierarchical tree structure. "
        #     f"Your task is to make the hierarchy shallower by {level} level(s) by promoting children nodes to become siblings with their parent nodes, "
        #     f"starting from the leaf nodes and working upwards (bottom-up approach).\n\n"
        #     f"SPECIFIC INSTRUCTIONS:\n"
        #     f"1. First identify all leaf nodes (nodes with no children)\n"
        #     f"2. Work upwards from these leaf nodes to find candidates for promotion\n"
        #     f"3. Prioritize nodes that have exactly one child - these are the best candidates for merging\n"
        #     f"4. When promoting a child node, it should become a sibling of its former parent\n"
        #     f"5. Preserve all node properties and only change the hierarchy structure\n"
        #     f"6. Never modify the root node\n\n"
        #     f"Here's the original tree structure in JSON format: "
        #     f"{load_json(hierarchy)}\n\n"
        #     f"CONSTRAINTS:\n"
        #     f"- Do not add or remove nodes\n"
        #     f"- Maintain all node data/content exactly as is\n"
        #     f"- The final tree height must be exactly {level} level(s) shallower than the original\n\n"
        #     f"PROCESS TO FOLLOW:\n"
        #     f"1. First calculate and show the current number of levels and a simple visualization of the hierarchy\n"
        #     f"2. Identify all leaf nodes and mark potential merge candidates (especially single-child nodes)\n"
        #     f"3. Perform the modifications starting from the leaves and moving upward\n"
        #     f"4. Verify the new tree height is reduced by exactly {level} level(s)\n"
        #     f"5. Return the modified tree structure in the same JSON format\n\n"
        #     f"OUTPUT REQUIREMENTS:\n"
        #     f"- Include a brief explanation of the changes made\n"
        #     f"- Show before/after level counts\n"
        #     f"- Return the final JSON structure"
        # )

        # prompt = (
        #     f"You are given a JSON representation of a hierarchical tree structure. "
        #     f"Your task is to make the hierarchy shallower by {level} level(s) by promoting children of a parent node to become siblings with the parent node that would reduce the tree height "
        #     f"starting from the leaf nodes and working upwards (bottom-up approach).\n\n"
        #     f"SPECIFIC INSTRUCTIONS:\n"
        #     f"1. First identify all leaf nodes (nodes with no children)\n"
        #     f"2. Work upwards from these leaf nodes to find candidates for promotion\n"
        #     f"3. Prioritize nodes that have exactly one child - these are the best candidates for merging\n"
        #     f"4. When promoting a child node, it should become a sibling of its former parent\n"
        #     f"5. Preserve all node properties and only change the hierarchy structure\n"
        #     f"6. Never modify the root node\n\n"
        #     f"Here's the original tree structure in JSON format: "
        #     f"{load_json(hierarchy)}"
        #     f"Do not add or remove nodes. "
        #     f"Return the modified tree structure in the same JSON format that reduces the height of the tree by {level} level(s)."
        #     f"Think step by step: first calculate the number of levels and the visualization of the hierarchy, then modify it and check whether the height has been reduced by {level} level(s)."
        # )

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Lower = more consistent translations
                )
                modified_hierarchy = response.choices[0].message.content
                return modified_hierarchy
            except Exception as e:
                print(f"OPENAI API Error: {e}")
                return modified_hierarchy


    def prune_hierarchy_graph(self, max_depth):
        """
        Prunes a hierarchy (tree) to reduce its depth by:
        - Flattening nodes with only one child and one parent.
        - Removing intermediate nodes that are not leaves and replacing them with direct links to their descendants.
        - Preserving all leaves.

        Parameters:
            G (nx.DiGraph): A directed graph representing the hierarchy.
            root (str): The root node of the hierarchy.
            max_depth (int): The maximum allowed depth.

        Returns:
            nx.DiGraph: The pruned hierarchy graph.
        """
        G = self.graph.copy()

        # 1. Flatten single-child chains (preserve leaves)
        for node in list(nx.topological_sort(G)):
            preds = list(G.predecessors(node))
            succs = list(G.successors(node))
            if len(preds) == 1 and len(succs) == 1:
                parent = preds[0]
                child = succs[0]
                if node != self.root and G.out_degree(node) == 1 and G.in_degree(node) == 1:
                    G.add_edge(parent, child)
                    G.remove_edge(parent, node)
                    G.remove_edge(node, child)
                    G.remove_node(node)

        # 2. Compute current depth
        def get_depths(G, root):
            depths = {}
            for node in nx.topological_sort(G):
                if node == root:
                    depths[node] = 0
                else:
                    preds = list(G.predecessors(node))
                    if preds:
                        depths[node] = min(depths[p] for p in preds) + 1
            return depths

        depths = get_depths(G, self.root)

        # 3. Remove intermediate nodes beyond depth if they are not leaves
        for node, depth in sorted(depths.items(), key=lambda x: -x[1]):
            if depth > max_depth and G.out_degree(node) > 0:
                preds = list(G.predecessors(node))
                succs = list(G.successors(node))
                for parent in preds:
                    for child in succs:
                        G.add_edge(parent, child)
                G.remove_node(node)

        self.graph = G

    def prune_hierarchy_graph_deepseek(self, max_depth):
        """
        Prunes a hierarchy (tree) to reduce its depth by:
        - Flattening nodes with only one child and one parent.
        - Removing intermediate nodes that are not leaves and replacing them with direct links to their descendants.
        - Preserving all leaves.

        Parameters:
            G (nx.DiGraph): A directed graph representing the hierarchy.
            root (str): The root node of the hierarchy.
            max_depth (int): The maximum allowed depth.

        Returns:
            nx.DiGraph: The pruned hierarchy graph.
        """
        G = self.graph.copy()

        # 2. Compute current depth
        def get_depths(G, root):
            depths = {}
            for node in nx.topological_sort(G):
                if node == root:
                    depths[node] = 0
                else:
                    preds = list(G.predecessors(node))
                    if preds:
                        depths[node] = min(depths[p] for p in preds) + 1
            return depths

        depths = get_depths(G, self.root)

        # 3. Remove intermediate nodes beyond depth if they are not leaves
        for node, depth in sorted(depths.items(), key=lambda x: -x[1]):
            if depth > max_depth and G.out_degree(node) > 0:
                preds = list(G.predecessors(node))
                succs = list(G.successors(node))
                for parent in preds:
                    for child in succs:
                        G.add_edge(parent, child)
                G.remove_node(node)

        # 1. Flatten single-child chains (preserve leaves)
        for node in list(nx.topological_sort(G)):
            preds = list(G.predecessors(node))
            succs = list(G.successors(node))
            if len(preds) == 1 and len(succs) == 1:
                parent = preds[0]
                child = succs[0]
                if node != self.root and G.out_degree(node) == 1 and G.in_degree(node) == 1:
                    G.add_edge(parent, child)
                    G.remove_edge(parent, node)
                    G.remove_edge(node, child)
                    G.remove_node(node)

        self.graph = G