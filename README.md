# LLM-guided-Hierarchy-Restructuring

PyTorch implementation of *Minimizing Hyperbolic Embedding Distortion with LLM-Guided Hierarchy Restructuring* paper.

## Hiererachy reformatting
The hierarchies that we used in this paper are all in the `./hierarchies` folder. The code that loads the hierarchies and modifies them is in the `hierarchy_utils.py`. The hierarchies go through 5 different steps that can be found in the `automatic_ontology_cleanup.py` file. Step 1 is generating the textual representation of the hierarchy, plus visualizing its degree histogram, the tree, and getting all the tree properties. In Step 2, the textual hierarchy representation goes through the LLM-guided restructuring process, either through API calls or directly using Chatbot LLMs such as DeepSeek and ChatGPT. Step 3, generates the new modified hierarchy in json format from the textual representation, if the hierarchy has been hallucinated, we will be informed in this step. We generate the hyperbolic embeddings using the new modified hierarchies in step 4 using `run_constructive_tree_embedding.py`. And once the embeddings are generated, the 5th step is evaluating the generated hierarchies through `evaluate.py`. The evaluation of bigger hierarchies is implemented in chunks in `distortion_memory_efficient` function to avoid OOM errors. The Visual Genome evaluation can run with a chunk size=100 on the local machine, and the ImageNet21-K ran on a 120GB server for 4 hours. You can adjust the chunk size based on your trade-off to be smaller and the run will take longer.

Since some of these hierarchies were big, DeepSeek could not handle the input size and instead returned a function to be applied on the hierarchy. You can run and check the function in `deepseek_bighierarchies.py`. The same holds for ChatGPT, you can find its function in `hierarchy_utils.py` in the `prune_hierarchy_graph` fuction.


The implementation of HS-DTE and Hadamard are from the following [code base](https://github.com/maxvanspengler/hyperbolic_tree_embeddings).
## Hierarchies
You can find all hierarchies: original, modified with ChatGPT and DeepSeek and all the hierarchies used in the ablation study in the `./hierarchies` folder.

## Embeddings
You can find all embeddings in the `./results` folder.

## Example: Embedding Trees
To get the Hadamard and HS-DTE embeddings of the hierarchies, run:

```sh
bash run_snellius.sh
```

## Evaluation
To get the Average distortion, Worst-case distortion, and MAP of the hyperbolic embeddings, run:

```sh
python evaluate.py
```

This calls the `evaluate_tree(tree, embeddings)` function.

## Dependencies
You can find the dependencies of our project in:
- `requirements_llm_guided.txt`

## License
This code is licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
