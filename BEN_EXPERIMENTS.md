# Experiment 1

## WITH standard dataset augmentations.

Commands:

Train on a 1 scene (1M steps).
```
./train.sh 0 cross_point_relative_1 online training.batch_size=1 training.epochs=1000000
```

Train on 10 scenes (1M steps).
```
./train.sh 0 cross_point_relative_10 online training.batch_size=10 training.epochs=1000000
```

Train on 100 scenes (1M steps). 7 batches per epoch.
```
./train.sh 0 cross_point_relative_100 online training.batch_size=16 training.epochs=142858
```

Train on 1000 scenes (1M steps). 63 batches per epoch.
```
./train.sh 0 cross_point_relative_1000 online training.batch_size=16 training.epochs=15873
```

Train on 10000 scenes (1M steps).
```
./train.sh 0 cross_point_relative_10000 online training.batch_size=16 training.epochs=1600
```

## WITHOUT standard dataset augmentations.

Commands:

Train on a 1 scene (1M steps).
```
./train.sh 0 cross_point_relative_1 online training.batch_size=1 training.epochs=1000000 dataset.anchor_transform_type=identity
```

Train on 10 scenes (1M steps).
```
./train.sh 1 cross_point_relative_10 online training.batch_size=10 training.epochs=1000000 dataset.anchor_transform_type=identity
```

Train on 100 scenes (1M steps). 7 batches per epoch.
```
./train.sh 0 cross_point_relative_100 online training.batch_size=16 training.epochs=142858 dataset.anchor_transform_type=identity
```

Train on 1000 scenes (1M steps). 63 batches per epoch.
```
./train.sh 1 cross_point_relative_1000 online training.batch_size=16 training.epochs=15873 dataset.anchor_transform_type=identity
```

Train on 10000 scenes (1M steps).
```
./train.sh 0 cross_point_relative_10000 online training.batch_size=16 training.epochs=1600 dataset.anchor_transform_type=identity
```


### Eval commands.

```
./eval.sh 0 a9v8wlzj dataset.data_dir=/home/beisner/datasets/cloth10k/cloth10k dataset.train_size=10000
