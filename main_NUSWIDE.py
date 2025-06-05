import os
import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
command = [
    "python", "UDLCH.py",
    "--data_name", "nuswide_fea",
    "--task_epochs", "50",
    "--train_batch_size", "64",
    "--category_split_ratio", "(10,0)",
    "--bit", "128",
    "--lr", "0.0001",

    "--alpha", "1",
    "--K", "2500",

    "--der_a", "20",
    "--der_b", "0.5",
    "--buffer_size", "2500",

    "--num_tasks", "2",
    "--max_epochs", "1",
    "--arch", "resnet50",
    "--num_hiden_layers", "3", "2",
    "--margin", "0.2",
    "--shift", "0.1",
    "--optimizer", "Adam",
    "--warmup_epoch", "5",
    "--pretrain",
]
subprocess.run(command)
