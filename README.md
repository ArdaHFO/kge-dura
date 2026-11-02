KGE-DURA is research software that predicts missing connections in collections of linked facts.

<h1 align="center">KGE-DURA</h1>

<p align="center"><strong>Regularization for knowledge graph completion.</strong></p>
<p align="center">Research code for duality-induced regularization of tensor-factorization knowledge graph embeddings.</p>

<p align="center">
  <strong>English</strong> · <a href="README.tr.md">Türkçe</a><br />
  <a href="https://arxiv.org/abs/2011.05816"><strong>Read the paper</strong></a> ·
  <a href="docs/REPRODUCING.md">Reproduction recipes</a> ·
  <a href="#citation">Cite this work</a>
</p>

---

This repository contains the research implementation accompanying **Duality-Induced Regularizer for Tensor Factorization Based Knowledge Graph Completion**, by **Zhanqiu Zhang, Jianyu Cai and Jie Wang**, published at **NeurIPS 2020**. It includes CP, ComplEx and RESCAL models, DURA regularizers, benchmark preprocessing and the original experiment commands.

The original authors and paper remain the attribution for the scientific contribution. This repository's bilingual presentation does not claim new authorship, newly reproduced results or a modernized implementation. [arXiv manuscript](https://arxiv.org/abs/2011.05816) · [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/f6185f0ef02dcaec414a3171cd01c697-Abstract.html)

## Start with the experiments

| Requirement | Historical environment documented by the project |
| --- | --- |
| Python | 3.6 or newer |
| PyTorch | 1.0–1.7 |
| NumPy | 1.17.2 or newer |
| tqdm | 4.41.1 or newer |
| Compute | CUDA-capable GPU and a matching CUDA-enabled PyTorch build |
| Data | WN18RR, FB15k-237 and YAGO3-10 splits included under `src_data/` |

These are the original dependency requirements, not a claim that every newer version is compatible. The training, evaluation and some regularizers call CUDA directly. A CPU-only execution path is not provided. There is no pinned requirements or environment file.

```bash
git clone https://github.com/ArdaHFO/kge-dura.git
cd kge-dura
```

Prepare an isolated environment compatible with the historical stack and your GPU. Then preprocess the bundled data from the `code` directory.

```bash
cd code
python process_datasets.py
```

The script writes mapped triples, entity/relation IDs and filtered-evaluation lookup data to `data/` at the repository root. Existing dataset output directories are skipped.

Choose the model/dataset combination in the [reproduction guide](docs/REPRODUCING.md). It preserves all nine original training commands and their hyperparameters. Run those commands from `code/`. `FB237` is the command-line name used for FB15k-237.

## Inside the implementation

- **Three tensor-factorization models.** CP, ComplEx and RESCAL share the knowledge graph completion interface in `models.py`.
- **Model-specific regularization.** DURA, weighted DURA and RESCAL variants are implemented alongside the existing comparison regularizers.
- **Reciprocal training triples.** The dataset loader augments triples with reversed relations for head and tail prediction.
- **Filtered ranking.** Evaluation reports mean reciprocal rank and Hits@1/3/10 while filtering known true triples.
- **Reproduction recipes.** Original settings cover each model on WN18RR, FB15k-237 and YAGO3-10.
- **Exported artifacts.** With `-save`, training writes configuration, logs, a checkpoint and entity/relation embeddings under `logs/`.

## Reported results

The original repository includes the following result figure for WN18RR, FB15k-237 and YAGO3-10. It is retained as the project's reported research result, not a new benchmark run performed for this documentation update.

<p align="center">
  <img src="result.png" alt="Original DURA benchmark results for WN18RR, FB15k-237 and YAGO3-10" />
</p>

See the paper for the scientific setting and interpretation. The original figure and experiment hyperparameters have not been changed.

## Development and validation

From the repository root, inspect available command-line options after installing the dependencies.

```bash
cd code
python learn.py --help
```

The repository does not include an automated test suite or CI workflow. The Python implementation and original instructions were inspected statically. Source fixes address batch dimensions, filtered-ranking mutation, dataset dimensions and CLI execution paths. No code, tests, preprocessing, training or benchmarks were run for these changes. The training entry point does not establish a fixed random seed, and compatibility with recent Python/PyTorch versions has not been validated here.

The supplied recipes retain `-save` to export logs, checkpoints and embeddings. Training without `-save` does not write these artifacts. Standalone evaluation uses `-test --checkpoint <run-directory>` with a matching dataset, model and rank. Review [`learn.py`](code/learn.py) before adapting the workflow. These corrected paths have not been validated by execution.

## Repository map

| Path | Purpose |
| --- | --- |
| [`code/process_datasets.py`](code/process_datasets.py) | Prepare benchmark splits and filtering structures |
| [`code/datasets.py`](code/datasets.py) | Load triples, add reciprocals and evaluate ranking |
| [`code/models.py`](code/models.py) | CP, ComplEx and RESCAL implementations |
| [`code/regularizers.py`](code/regularizers.py) | DURA and other regularizers |
| [`code/optimizers.py`](code/optimizers.py) | Minibatch optimization loop |
| [`code/learn.py`](code/learn.py) | Training arguments, evaluation and artifact export |
| [`src_data/`](src_data/) | Bundled benchmark splits and dataset notices |
| [`docs/REPRODUCING.md`](docs/REPRODUCING.md) | Preserved original experiment commands |
| [`result.png`](result.png) | Original reported result figure |

## Contributing

For reproducibility reports, include the commit, environment, CUDA/PyTorch versions, dataset, complete command and observed metrics. Explain any changes to seeds, preprocessing or hyperparameters. Preserve the original recipes and distinguish new experimental results from those reported by the paper. Documentation and portability improvements should retain the authors' attribution and dataset notices.

## Citation

If this code is useful in your research, cite the original paper.

```
@inproceedings{NEURIPS2020_f6185f0e,
 author = {Zhang, Zhanqiu and Cai, Jianyu and Wang, Jie},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {21604--21615},
 publisher = {Curran Associates, Inc.},
 title = {Duality-Induced Regularizer for Tensor Factorization Based Knowledge Graph Completion},
 url = {https://proceedings.neurips.cc/paper/2020/file/f6185f0ef02dcaec414a3171cd01c697-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

## License and credits

The original implementation acknowledges [facebookresearch/kbc](https://github.com/facebookresearch/kbc). The authors also provide the related work **Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction**, by Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang and Jie Wang, AAAI 2020. [Paper](https://arxiv.org/pdf/1911.09419.pdf) · [KGE-HAKE code](https://github.com/MIRALab-USTC/KGE-HAKE)

No project-level software license file is present in this repository. Dataset notices are retained separately in [FB237](src_data/FB237/license.docx), [WN18RR](src_data/WN18RR/license.txt) and [YAGO3-10](src_data/YAGO3-10/license.txt). The WN18RR notice refers to `WN/license.txt`, which is not included in this tree. The YAGO3-10 notice identifies CC-BY for the source dumps. These dataset notices do not establish a license for the entire codebase, and this documentation assigns no new license.
