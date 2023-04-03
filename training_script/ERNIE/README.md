# Fine-tuning ERNIE on GLUE

## Setup

1. Install NVIDIA NCCL

    This repo requires NCLL library, please see [here](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html) for installaton instruction.
    
1. Install PaddlePaddle

    ERNIE requires PaddlePaddle 1.7.0+, please see [here](https://www.paddlepaddle.org.cn/install/quick) for installaton instruction.
    
1. Install requirements

    ```
   pip install -r requirements.txt
   ```

## Download data and models

Run the following command to download GLUE dataset and ERNIE models.

```
bash data/download_data_and_models.sh
```

## Fine-tuning

Run all the scripts in folder ```./scripts```. For example, 

```
bash scripts/run_base_cola.sh
```

Each script is named in the form: 

```
run_{ERNIE_MODEL}_{TASK_NAME}.sh
```

where ```ERNIE_MODEL``` specifies the model we use (ERNIE-base or ERNIE-large) and ```TASK_NAME``` specifies a task in GLUE Benchmark.
