# Experiment Results

## Config Overview
```yaml
# config/default.yaml
#

experiments:
  # “smoke‐tests”
  - experiment_name: "smoke_test_cpu"
    dataset: "Cora"
    model: "gcn"
    epochs: 5
    batch_size: [16, 32, 64, 128, 256, 512]
    lr: 0.001
    hidden_dim: 64
    seed: 42
    world_size: 1
    no_cuda: true

  - experiment_name: "smoke_test_gpu"
    dataset: "Cora"
    model: "gcn"
    epochs: 5
    batch_size: [16, 32, 64, 128, 256, 512]
    lr: 0.001
    hidden_dim: 64
    seed: 42
    world_size: 1
    no_cuda: false
```

## smoke_test_cpu
**Different config:** no_cuda=true

| Batch Size | Val Acc | Throughput |
|:----------:|:-------:|:----------:|
| 16 | - | - |
| 32 | - | - |
| 64 | - | - |
| 128 | - | - |
| 256 | - | - |
| 512 | - | - |

## smoke_test_gpu
**Different config:** no_cuda=false

| Batch Size | Val Acc | Throughput |
|:----------:|:-------:|:----------:|
| 16 | - | - |
| 32 | - | - |
| 64 | - | - |
| 128 | - | - |
| 256 | - | - |
| 512 | - | - |

## Experiment Metadata from the Database
_No runs recorded yet._
