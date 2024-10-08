# VLM_SS 
Mini-batch selective sampling for knowledge adaption of VLMs

Knowledge-grounded Adaptation Strategy for Vision-language Models: Building a Unique Case-set for Screening Mammograms for Residents Training [MICCAI 2024]
[Aisha Urooj Khan](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=ceiuCp4AAAAJ), [John Garrett](https://scholar.google.com/citations?user=v6ZI4NEAAAAJ&hl=en), [Tyler Bradshaw](https://scholar.google.com/citations?hl=en&user=NaqaiKoAAAAJ), [Lonie Salkowski](https://www.uwhealth.org/providers/lonie-r-salkowski-md), [Jiwoong Jeong](https://scholar.google.com/citations?hl=en&user=rGLrad0AAAAJ), [Amara Tariq](https://scholar.google.com/citations?hl=en&user=3_Evjy4AAAAJ), [Imon Banerjee](https://scholar.google.com/citations?hl=en&user=hagJ_W8AAAAJ)

[`Paper`](https://arxiv.org/abs/2405.19675) | [`BibTeX`](#citation)

Official Pytorch implementation of Mini-batch selective sampling for knowledge adaption of VLMs and pre-trained models.

### Requirements
The code for selective sampling works for python 3.x versions and requires the following packages: 
random, math, and tqdm

For selective sampling in [ALBEF](https://github.com/salesforce/ALBEF), we use the environment provided from ALBEF original repo, and update/install a few packages using [ALBEF/requirements.txt](https://github.com/aurooj/VLM_SS/blob/main/ALBEF/requirements.txt).

For selective sampling in [MedCLIP](https://github.com/RyanWangZf/MedCLIP), we use the environment provided in [MedCLIP/requirements.txt](https://github.com/aurooj/VLM_SS/blob/main/MedCLIP/requirements.txt). 

### Usage
Mini-batch selective sampling requires the mammography image-report pairs to have the `group` information. `group` information can be extracted from radiology reports using the [notebook](https://github.com/aurooj/VLM_SS/blob/main/extract_groups.ipynb) `extract_groups.ipynb` provided in this code repo. 

Assuming that your data is in the format as provided in `sample_data.json`, you can enable the selective sampling by creating a `SelectiveSampling` object in your custom Pytorch Dataset class as follows:
1. Import selective sampling in your custom dataset code:
```
from selective_sampling import SelectiveSampling
```

2. Add selective sampling to your custom dataset class in the `__init__()` function:
```
self.selective_sampling = SelectiveSampling(data)
```
where `data` is the list of data instances' dictionaries

Each dictionary in this list belongs to a data instance, and
            can have an arbitrary number of columns but expects the key 'group' in each item's dictionary.
            Example:
       ```
               Format: [
                { 'colA':<value>,
                'colB':<value>,
                'group':['a', 'b']
                }, 
                { 'colA':<value>,
                'colB':<value>,
                'group':['a', 'b']
                }, 
                ...
                ]
        ```

3. Add `shuffle()` definition to your dataset class:
```
def shuffle(self, bs=8, rare_grp_ratio=0.375, batch_shuffle=False):
        #function to prepare minibatches based on selective sampling strategy
        # calls shuffle function from the base class (SelectiveSampling) 
        self.ann = self.selective_sampling.shuffle(bs=bs, rare_grp_ratio=rare_grp_ratio, batch_shuffle=batch_shuffle)
```
   
where ```bs```=batch_size, ```rare_grp_ratio```=ratio of samples from rare groups in a mini-batch, ```batch_shuffle```= a boolean flag, if set to True, mini-batch is shuffled further; default value is set to `False` based on the ablations reported in table 3, rows (5) and (6) in the paper. 

4. Now in your main training loop, you can call the selective sampling based shuffling by calling the shuffle method in your custom dataset class, i.e., `data_loader.dataset.shuffle(bs=<batch_size>)`.
   
   ```
   for epoch in range(epochs):
        
        if config['selective_sampling']:
            data_loader.dataset.shuffle(bs=config['batch_size'])
   ```
   See example snippet from `ALBEF/Pretrain.py` below:

   
```
        for epoch in range(start_epoch, max_epoch):

                if config['selective_sampling']:
                    #at every epoch, shuffle data with custom sampling function for medical data
                    print(f"Shuffling training data for epoch {epoch}")
                    data_loader.dataset.shuffle(bs=config['batch_size'],
                                                rare_grp_ratio=config['rare_grp_ratio'],
                                                batch_shuffle=config['batch_shuffle']
                                                )
                        ...
```


  #### NOTE: 
The default sampling from PyTorch, i.e., `data_loader.shuffle` should be set to `False` in the training dataloader when using selective sampling based shuffling. 

We integrated selective sampling as part of this work into ALBEF and MedCLIP code repos. The updated codes are provided as part of this code repo. We thank the authors of ALBEF and MedCLIP for providing their amazing code repos. 

### ALBEF with Selective Sampling
Read instructions from [albef.md](https://github.com/aurooj/VLM_SS/blob/main/albef.md) for ALBEF trained with selective sampling. 

### MedCLIP with Selective Sampling
Read instructions from [medclip.md](https://github.com/aurooj/VLM_SS/blob/main/medclip.md) for MedCLIP trained with selective sampling. 

### Model Checkpoints
See [this](https://github.com/aurooj/VLM_SS/blob/main/ALBEF/pretrained_models/readme.md) to download model checkpoints for our best pretrained model and retrieval model from ALBEF.

##### Todo: 
- [x] Add MedCLIP's modified version with selective sampling here.
- [x] Add pretrained model weights

### Citation
If this work and/or its findings are useful for your research, please cite our paper.
Todo: 
- [ ] Replace with bibtex from MICCAI

```bibtex
@misc{khan2024knowledgegrounded,
    title={Knowledge-grounded Adaptation Strategy for Vision-language Models: Building Unique Case-set for Screening Mammograms for Residents Training},
    author={Aisha Urooj Khan and John Garrett and Tyler Bradshaw and Lonie Salkowski and Jiwoong Jason Jeong and Amara Tariq and Imon Banerjee},
    year={2024},
    eprint={2405.19675},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

### Questions?
Please contact 'aishaurooj@gmail.com'




