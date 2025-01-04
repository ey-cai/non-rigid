# Instructions

When running on autobot, you can do something like this:

```bash
./cluster/launch_autobot.sh -t rtx3090 -n 1 'python scripts/train.py --config-name commands/dedo/hangproccloth_multimodal/cross_flow_relative/train dataset.data_dir=/project_data/held/baeisner/tax3d_data/proccloth'
```

where the command is fully single-quote wrapped and the dataset path is set to the correct location.
