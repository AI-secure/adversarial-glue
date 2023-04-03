# Fine-tuning DeBERTa on GLUE

## Setup

1. Install PyTorch

    This repo requires PyTorch, please see [here](https://pytorch.org/get-started/locally/#start-locally) for installaton instruction.
    
1. Install requirements

    ```
   pip install -r requirements.txt
   ```

## Fine-tuning

Run all the scripts in folder ```./scripts```. For example, 

```
bash scripts/run_large_cola.sh
```

Each script is named in the form: 

```
run_{DeBERTa_MODEL}_{TASK_NAME}.sh
```

where ```DeBERTa_MODEL``` specifies the model we use (DeBERTa-Large or DeBERTa-V2-XXLarge) and ```TASK_NAME``` specifies a task in GLUE Benchmark.
