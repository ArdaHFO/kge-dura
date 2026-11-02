# Reproducing the published DURA experiments

[English overview](../README.md) · [Türkçe genel bakış](../README.tr.md)

The following preprocessing instructions and all nine experiment recipes are preserved from this repository's original README. They are historical reproduction settings for the paper by Zhanqiu Zhang, Jianyu Cai and Jie Wang, not newly executed results.

Use the historical dependency stack described in the overview, with CUDA-enabled PyTorch. Run training from `code/` because the scripts resolve `../src_data`, `../data` and `../logs` relative to the current directory. `CUDA_VISIBLE_DEVICES` below uses POSIX shell syntax. Choose an available GPU on your own machine. The original WN18RR RESCAL example selects GPU 2.

Preprocessing writes into `data/`. Existing dataset output directories are skipped by the script, so preserve or inspect previous outputs before attempting a fresh preprocessing run. The original recipes retain `-save` to export logs, checkpoints and embeddings. Training without `-save` writes no training log or model artifacts.

## Reproduce the Results

### 1. Preprocess the Datasets
To preprocess the datasets, run the following commands.

```shell script
cd code
python process_datasets.py
```

Now, the processed datasets are in the `data` directory.

### 2. Reproduce the Results 
To reproduce the results of CP, ComplEx and RESCAL with
the DURA regularizer on WN18RR, FB15k237 and YAGO3-10,
please run the following commands.

```shell script
#################################### WN18RR ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model CP --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer DURA --reg 1e-1 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model ComplEx --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 1e-1 --max_epochs 50 \
--valid 5 -train -id 0 -save -weight

# RESCAL
CUDA_VISIBLE_DEVICES=2 python learn.py --dataset WN18RR --model RESCAL --rank 256 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1024 --regularizer DURA_RESCAL --reg 1e-1 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

#################################### FB237 ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model CP --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model ComplEx --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save

# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 512 --regularizer DURA_RESCAL --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save


#################################### YAGO3-10 ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model CP --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1000 --regularizer DURA_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model ComplEx --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1000 --regularizer DURA_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save

# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1024 --regularizer DURA_RESCAL_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight
```


## Outputs and scope

With `-save`, the run directory under `logs/` contains the argument configuration, training log, model checkpoint and exported embeddings. The experiment names include model, regularizer, dataset and model ID.

The source computes filtered head- and tail-prediction MRR and Hits@1/3/10. Reproducing the reported numbers requires the full training/evaluation workflow. No experiment was run as part of the documentation refresh. The repository does not pin a complete environment or set a fixed random seed in the training entry point.

Standalone evaluation uses `-test --checkpoint <run-directory>` with the matching dataset, model and rank. The checkpoint directory must contain the saved `checkpoint` file. See [`code/learn.py`](../code/learn.py). The static fixes to argument handling and ranking have not been exercised by a new training or evaluation run.
