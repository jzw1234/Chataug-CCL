-----

## Dependencies
Main libraries
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.7 +
- [transformers](https://github.com/huggingface/transformers) 4.21.0
- [fastNLP](https://github.com/fastnlp/fastNLP) 1.0.0beta
```
pip install transformers == 4.21.0
pip install fastNLP == 1.0.0beta
```

	
All code only supports running on Linux.


### data
we have prepared the raw dataset to help you reproduce the results in our paper.  Datasets provided by this repo can  **only**  be used for *Reproduction* and *Research Purposes*.
All files are in `jsonl` format where each line is a `json` sample:
```
{"source": "the input text (for encoder)", "target": "the target text (for decoder)"}}
```
(We will provide the enhanced radiology reports datasets via ChatGPT.)

Before loading the training set, please pre-tokenize these files  with the following command:
```
mkdir jsonl_files
mkdir tokenized_files
mv /download/path/rsum.zip  ./jsonl_files
cd jsonl_files
unzip rsum.zip && cd ..
python preprocess/preprocess.py --model_name  t5 --dataset rsum
``` 


### Training
We have provided the training scripts for each dataset we used in this paper, and you can easily start the training process with them:

```
#If there is no warmed-up checkpoint, you should use `--warmup True` to train the generation model with NLLLoss 
python run_rsum.py --mode train --gpus 0 --warmup True --model_name t5
```

the warmed-up checkpoint will be saved to `./pretrained_weigths/rsum/t5(or pegasus)` by default.  
Please notice that huggingface also provides many finetuned checkpoints.

```

After completing the training process,  several best checkpoints will be stored in a folder named after the training start time by default, for example, `checkpoints/rsum/t5/2022-10-05-10-37-24-196200`

### Generation
```
python run_rsum.py --mode val (or test) --model_name t5 --save_path checkpoints/rsum/t5/2022-10-05-10-37-24-196200/ --gpus 0
```
To generate the results for test set with  **a specified checkpoint**, you can use the `--ckpt`  parameter and remember to change the mode to `test`:
```
python run_rsum.py --mode test --model_name t5-small --save_path checkpoints/rsum/t5/2022-10-05-10-37-24-196200/ \
--ckpt epoch-2_step-8000.pt --gpus 0,1,2,3
```
This will produce the generated results in the floder `results/rsum/t5/2022-10-05-10-37-24-196200/`  containing `epoch-2_step-8000.test.sys` , `epoch-2_step-8000.test.ref`

### Evaluation
We have proveded the evaluation scripts for each dataset: `evaluation/$dataset/eval.py` with which you can easily get the evaluation results.

This is an example to evaluate all the generated results for `rsum` in the folder `results/rsum/t5/2022-10-05-10-37-24-196200/`:
```
python evaluation/rsum/eval.py --sys_path results/rsum/t5/2022-10-05-10-37-24-196200/
```
If you only want to evaluate a specified file：
```
python evaluation/rsum/eval.py --sys_file results/rsum/t5/2022-10-05-10-37-24-196200/epoch-2_step-8000.sys
```
