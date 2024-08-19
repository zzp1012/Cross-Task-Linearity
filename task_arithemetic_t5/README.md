# Editing Models with Task Arithmetic

This directory contains our implementation of T5 model for the ICLR 2023 paper [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089). **`cross-task-linearity.py` contains the main implementation of cross-task-linearity.**

You should download the dataset from [huggingface](https://huggingface.co/) including [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb/), [QASC](https://huggingface.co/datasets/allenai/qasc) and etc. The fine-tuned T5 model can be downloaded from https://huggingface.co/mrm8488


Here are the scripts to run the Cross-Task-Linearity Evaluation:

```bash
python cross-task-linearity.py \
            --save_root ./outs/cml_addition/qasc \
            --modelA_path "t5-base-finetuned-qasc" \
            --modelB_path "t5-base-finetuned-imdb" \
            --modelA_coef 0.8 \
            --modelB_coef 0.8 \
            --sample_num 500 \
            --dataset qasc
```