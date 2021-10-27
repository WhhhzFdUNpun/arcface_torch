from pathlib import Path

from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = False
config.output = f"/output/{Path(__file__).resolve().stem}"

config.dataset = "webface"
config.embedding_size = 2048
config.sample_rate = 1
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 32
config.lr = 1  # batch size is 512

config.rec = "/data"
config.num_classes = 10572
config.num_image = "forget"
config.num_epoch = 20
config.warmup_epoch = -1
config.decay_epoch = [20, 30, 40]
config.val_targets = ["agedb_30", "dev_00", "dev_04"]
