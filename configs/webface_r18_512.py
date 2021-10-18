from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r18"
config.resume = False
config.output = "/output/webface_r18_512"

config.dataset = "webface"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512

config.rec = "/data"
config.num_classes = 10572
config.num_image = "forget"
config.num_epoch = 1  # było 34
config.warmup_epoch = -1
config.decay_epoch = [20, 28, 32]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
