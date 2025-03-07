# NeuroDx-LM

## Environment Set Up

```md
pip install -r requirements.txt
```

## Datasets and Preprocessing

Download the dataset from the following two links and use the methods in `dataset_maker` to process the dataset.

```md
- [CHB](https://physionet.org/content/chbmit/1.0.0/)
- [Schizophrenia](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441)
```

## Run Experiments

### Load Pretrained Weights

Obtain the pretrained weights from this repository and place them in the `checkpoints` folder.

```md
- [Labram](https://github.com/935963004/LaBraM)
```

### Load First-Stage Training Weights

The first-stage training weights can be obtained from the `checkpoints` directory in this repository.


### Run the Model

```md
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=1 run_finetuning.py \
    --output_dir \
    --log_dir \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --weight_decay 0.05 \
    --batch_size 64 \
    --lr 1e-4 \
    --update_freq 1 \
    --warmup_epochs 3 \
    --epochs 20 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --dist_eval \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset \
    --disable_qkv_bias \
    --seed 0 \
    --stage_2 \
    --use_lora \
    --lora_1_path 
```

