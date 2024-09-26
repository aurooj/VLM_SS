## ALBEF with Selective Sampling

#### Pretrain

For pretraining, run the following commands:

```
cd ALBEF/
```

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env Pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain 
```

Check `./configs/Pretrain.yaml` for selective sampling settings.


#### Finetuning for Image-to-Report and Report-to-Image Retrieval 
For retrieval task, run the following command:

```
python -m torch.distributed.launch --nproc_per_node=1 --use_env Retrieval.py \
--config ./configs/Retrieval_coco.yaml \
--output_dir output/Retrieval \
--checkpoint [Pretrained checkpoint]

```

Check `./configs/Retrieval_coco.yaml` for selective sampling settings.
