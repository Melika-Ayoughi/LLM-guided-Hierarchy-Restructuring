import re
from collections import defaultdict
import pprint

def load_hierarchy(file_path):
    """Load the hierarchy from a text file and return as a nested dictionary."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    stack = []
    root = {}
    current_level = -1

    for line in lines:
        # Determine the level by counting leading spaces/tabs
        level = (len(line) - len(line.lstrip())) // 2

        # Get the node ID
        node_id = line.strip().split('-')[-1].strip()

        # Update stack based on current level
        while len(stack) > level:
            stack.pop()

        # Add to the hierarchy
        if not stack:
            root[node_id] = {}
            stack.append((node_id, root[node_id]))
        else:
            parent_id, parent_dict = stack[-1]
            parent_dict[node_id] = {}
            stack.append((node_id, parent_dict[node_id]))

    return root


def analyze_depth(hierarchy, current_depth=0, depth_info=None):
    """Analyze depth distribution of the hierarchy."""
    if depth_info is None:
        depth_info = defaultdict(int)

    if not hierarchy:
        depth_info[current_depth] += 1
        return depth_info

    for child in hierarchy.values():
        analyze_depth(child, current_depth + 1, depth_info)

    return depth_info

def flatten_deep_branches(hierarchy, max_depth, current_depth=0):
    """Flatten branches that exceed max_depth by promoting all descendants to this level."""
    new_hierarchy = {}

    for node_id, children in hierarchy.items():
        if current_depth >= max_depth:
            # Promote this node and ALL its descendants to this level
            new_hierarchy[node_id] = {}
            all_descendants = {}
            _collect_all_descendants(children, all_descendants)
            new_hierarchy.update(all_descendants)
        else:
            # Keep structure and recurse
            new_hierarchy[node_id] = flatten_deep_branches(children, max_depth, current_depth + 1)

    return new_hierarchy


def _collect_all_descendants(hierarchy, flat_dict):
    """Collect all descendants into a flat dictionary (ignoring structure)."""
    for node_id, children in hierarchy.items():
        flat_dict[node_id] = {}
        _collect_all_descendants(children, flat_dict)

def collapse_single_child_chains(hierarchy):
    """Collapse chains where nodes have only one child into a single merged node."""
    new_hierarchy = {}

    for node_id, children in hierarchy.items():
        chain = [node_id]
        current = children

        # Walk down single-child chain
        while len(current) == 1:
            next_node, next_children = next(iter(current.items()))
            chain.append(next_node)
            current = next_children

        # Recurse on remaining subtree
        collapsed_subtree = collapse_single_child_chains(current)
        merged_node_id = '_'.join(chain)
        new_hierarchy[merged_node_id] = collapsed_subtree

    return new_hierarchy

# def flatten_deep_branches(hierarchy, max_depth, current_depth=0):
#     """Flatten branches that exceed max_depth by promoting nodes."""
#     new_hierarchy = {}
#
#     for node_id, children in hierarchy.items():
#         if current_depth >= max_depth:
#             # We're at max depth, promote all descendants to this level
#             new_hierarchy[node_id] = {}
#             # Recursively add all descendants at this level
#             _promote_descendants(children, new_hierarchy[node_id])
#         else:
#             # Keep this node and process its children
#             new_hierarchy[node_id] = flatten_deep_branches(
#                 children, max_depth, current_depth + 1
#             )
#
#     return new_hierarchy
#
#
# # def _promote_descendants(children, target_dict):
# #     """Helper function to promote all descendants to target level."""
# #     for child_id, grand_children in children.items():
# #         target_dict[child_id] = {}
# #         _promote_descendants(grand_children, target_dict[child_id])
# def _promote_descendants(children, target_dict):
#     """Promote all descendants to the same level by flattening structure."""
#     for child_id, grand_children in children.items():
#         target_dict[child_id] = {}  # Promote to current level
#         _promote_descendants(grand_children, target_dict)  # Keep flattening to this level

# def collapse_single_child_chains(hierarchy):
#     """Collapse linear chains where nodes have only one child."""
#     new_hierarchy = {}
#
#     for node_id, children in hierarchy.items():
#         # Process children first
#         processed_children = collapse_single_child_chains(children)
#
#         # If this node has only one child, and that child has only one child, etc.
#         # We'll collapse the entire chain
#         if len(processed_children) == 1:
#             child_id, grand_children = next(iter(processed_children.items()))
#             new_node_id = f"{node_id}_{child_id}"  # Combine IDs to preserve information
#             new_hierarchy[new_node_id] = grand_children
#         else:
#             new_hierarchy[node_id] = processed_children
#
#     return new_hierarchy
# def collapse_single_child_chains(hierarchy):
#     """Collapse chains where nodes have only one child by merging them into a single node."""
#     new_hierarchy = {}
#
#     for node_id, children in hierarchy.items():
#         current_id = node_id
#         current_children = children
#
#         # Collapse while there's exactly one child
#         while len(current_children) == 1:
#             child_id, grand_children = next(iter(current_children.items()))
#             current_id += f"_{child_id}"
#             current_children = grand_children
#
#         # Recurse into remaining children
#         new_hierarchy[current_id] = collapse_single_child_chains(current_children)
#
#     return new_hierarchy

def optimize_hierarchy(file_path):
    """Main function to load and optimize the hierarchy."""
    # 1. Load the hierarchy
    hierarchy = load_hierarchy(file_path)

    # 2. Analyze depth distribution
    depth_info = analyze_depth(hierarchy)
    print("Original depth distribution:", dict(depth_info))

    # Determine the depth threshold (deepest 20%)
    if depth_info:
        max_original_depth = max(depth_info.keys())
        depth_threshold = int(max_original_depth * 0.8)
        print(f"Will flatten branches deeper than {depth_threshold} levels")
    else:
        depth_threshold = 5  # default if no depth info

    # 3. Flatten deep branches
    flattened = flatten_deep_branches(hierarchy, depth_threshold)

    # 4. Collapse single-child chains
    optimized = collapse_single_child_chains(flattened)

    # Verify no multiple inheritance exists (structure remains a tree)
    # Our operations maintain tree structure, so no need to check

    # 5. Convert back to text format
    def dict_to_text(hier, level=0):
        lines = []
        for node_id, children in hier.items():
            # Split combined nodes back for display (but keep the structure)
            display_ids = node_id.split('_')
            for i, display_id in enumerate(display_ids):
                indent = '  ' * (level + i)
                lines.append(f"{indent}- {display_id}")
            lines.extend(dict_to_text(children, level + len(display_ids)))
        return lines

    optimized_text = '\n'.join(dict_to_text(optimized))
    print("\nFlattened hierarchy:")
    pprint.pprint(flattened)

    print("\nCollapsed hierarchy:")
    pprint.pprint(optimized)

    print("\nOptimized text:")
    print(optimized_text)

    return optimized_text
# #
# import re
# from collections import defaultdict
#
#
# def load_hierarchy(file_path):
#     """Load the hierarchy from a text file and return as a nested dictionary."""
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#
#     stack = []
#     root = {}
#     current_level = -1
#
#     for line in lines:
#         # Determine the level by counting leading spaces/tabs
#         level = (len(line) - len(line.lstrip())) // 2
#
#         # Get the node ID
#         node_id = line.strip().split('-')[-1].strip()
#
#         # Update stack based on current level
#         while len(stack) > level:
#             stack.pop()
#
#         # Add to the hierarchy
#         if not stack:
#             root[node_id] = {}
#             stack.append((node_id, root[node_id]))
#         else:
#             parent_id, parent_dict = stack[-1]
#             parent_dict[node_id] = {}
#             stack.append((node_id, parent_dict[node_id]))
#
#     return root
#
#
# def analyze_depth(hierarchy, current_depth=0, depth_info=None):
#     """Analyze depth distribution of the hierarchy."""
#     if depth_info is None:
#         depth_info = defaultdict(int)
#
#     if not hierarchy:
#         depth_info[current_depth] += 1
#         return depth_info
#
#     for child in hierarchy.values():
#         analyze_depth(child, current_depth + 1, depth_info)
#
#     return depth_info
#
#
# def flatten_deep_branches(hierarchy, max_depth, current_depth=0):
#     """Flatten branches that exceed max_depth by promoting nodes."""
#     new_hierarchy = {}
#
#     for node_id, children in hierarchy.items():
#         if current_depth >= max_depth:
#             # We're at max depth, promote all descendants to this level
#             # Instead of keeping the structure, we'll add all descendants directly
#             new_hierarchy[node_id] = {}
#             _collect_descendants(children, new_hierarchy[node_id])
#         else:
#             # Keep this node and process its children
#             new_hierarchy[node_id] = flatten_deep_branches(
#                 children, max_depth, current_depth + 1
#             )
#
#     return new_hierarchy
#
#
# def _collect_descendants(children, target_dict):
#     """Actually promote all descendants to become siblings at the current level"""
#     for child_id, grand_children in children.items():
#         # Add the child to the current level
#         target_dict[child_id] = {}
#         # Instead of nesting, we promote all grandchildren to be children of the current level
#         # Essentially making former grandchildren become siblings
#         for grandchild_id, great_grand_children in grand_children.items():
#             target_dict[grandchild_id] = {}
#             # Continue promoting all levels below
#             _collect_descendants(great_grand_children, target_dict[grandchild_id])
#
#
# def collapse_single_child_chains(hierarchy):
#     """Actually collapse linear chains by making grandchildren direct children"""
#     new_hierarchy = {}
#
#     for node_id, children in hierarchy.items():
#         # First process all children recursively
#         processed_children = collapse_single_child_chains(children)
#
#         # If this node has exactly one child
#         if len(processed_children) == 1:
#             child_id, grand_children = next(iter(processed_children.items()))
#             # Make all grandchildren direct children of current node
#             new_hierarchy[node_id] = grand_children
#             # Combine the IDs to preserve information
#             new_hierarchy[f"{node_id}_{child_id}"] = grand_children
#         else:
#             new_hierarchy[node_id] = processed_children
#
#     return new_hierarchy
#
#
# def dict_to_text(hier, level=0):
#     """Convert the hierarchy dictionary back to text format."""
#     lines = []
#     for node_id, children in hier.items():
#         # Split combined nodes for display
#         display_ids = node_id.split('_')
#         for i, display_id in enumerate(display_ids):
#             indent = '  ' * (level + i)
#             lines.append(f"{indent}- {display_id}")
#         lines.extend(dict_to_text(children, level + len(display_ids)))
#     return lines
#
#
# def optimize_hierarchy(file_path):
#     """Main function to load and optimize the hierarchy."""
#     # 1. Load the hierarchy
#     hierarchy = load_hierarchy(file_path)
#
#     # 2. Analyze depth distribution
#     depth_info = analyze_depth(hierarchy)
#     print("Original depth distribution:", dict(depth_info))
#
#     # Determine the depth threshold (deepest 20%)
#     if depth_info:
#         max_original_depth = max(depth_info.keys())
#         depth_threshold = int(max_original_depth * 0.8)
#         print(f"Will flatten branches deeper than {depth_threshold} levels")
#     else:
#         depth_threshold = 5  # default if no depth info
#
#     # 3. Flatten deep branches
#     flattened = flatten_deep_branches(hierarchy, depth_threshold)
#
#     # 4. Collapse single-child chains
#     optimized = collapse_single_child_chains(flattened)
#
#     # 5. Convert back to text format
#     optimized_text = '\n'.join(dict_to_text(optimized))
#     return optimized_text

# Example usage:
# optimized_hierarchy = optimize_hierarchy('your_hierarchy.txt')
# print(optimized_hierarchy)
# with open('optimized_hierarchy.txt', 'w') as f:
#     f.write(optimized_hierarchy)

# # Example usage:
# optimized_hierarchy = optimize_hierarchy('./hierarchies/Visual_Genome/Visual_Genome.txt')
# # print(optimized_hierarchy)
# with open('./hierarchies/Visual_Genome/Visual_Genome_optimized/Visual_Genome_deepseek_corrected.txt', 'w') as f:
#     f.write(optimized_hierarchy)
#
#
# optimized_hierarchy = optimize_hierarchy('./hierarchies/Imagenet21k_v1/Imagenet21k_v1.txt')
# # print(optimized_hierarchy)
# with open('./hierarchies/Imagenet21k_v1/Imagenet21k_v1_optimized/Imagenet21k_v1_deepseek_corrected.txt', 'w') as f:
#     f.write(optimized_hierarchy)

hierarchy_names = [
        # "cub_three_level",
        "ImageNet-1k",
        "pizza",
        # "Core50_unbalanced",
        # "Core50",
        # "Madverse",
        "Matador",
        "moments_in_time",
        # "BioTrove-LifeStages",
        "marine-tree",
        "NABirds",
        # "COCO10k",
        "EgoObjects",
        # "OpenLoris",
        "PascalVOC",
        # "Imagenet21k_v1",
        # "Visual_Genome"
        # # # "BioTrove-Balanced",
        # # # "BioTrove-Unseen",
        # # # "rare_species",
        # # # "Caltech101",
    ]
for hierarchy_name in hierarchy_names:
    optimized_hierarchy = optimize_hierarchy(f'./hierarchies/{hierarchy_name}/{hierarchy_name}.txt')
    # print(optimized_hierarchy)
    new_adr = f'./hierarchies/{hierarchy_name}/{hierarchy_name}_optimized/{hierarchy_name}_deepseek/{hierarchy_name}_deepseek.txt'
    with open(new_adr, 'w') as f:
        f.write(optimized_hierarchy)

    new_hierarchy = load_hierarchy(new_adr)
    depth_info = analyze_depth(new_hierarchy)
    print("Optimized depth distribution:", dict(depth_info))