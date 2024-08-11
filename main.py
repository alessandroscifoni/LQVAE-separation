from jukebox.data.data_processor import DataProcessor
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
import math
import os
from jukebox.hparams import setup_hparams
from jukebox.make_models import make_vqvae, make_prior, restore_opt, save_checkpoint
from jukebox.utils.logger import init_logging

from jukebox.utils.torch_utils import zero_grad, count_parameters
from jukebox.utils.dist_utils import print_once, allreduce, allgather
from jukebox.train import *

from jukebox.data.data_processor import DataProcessor
hps = "vqvae"
hps = setup_hparams(hps, dict(sample_length=1048576))
hps.audio_files_dir=os.getcwd()+"/babyslakh_16k/"
hps.test_audio_files_dir=os.getcwd()+"/babyslakh_16k/"
hps.channels=2
hps.sample_length=1048576
hps.min_duration = math.ceil(hps.sample_length / hps.sr)
hps.max_duration = math.inf
hps.labels = False


# Print the current working directory
print("Current Working Directory:", os.getcwd())
data_processor = DataProcessor(hps)
# Setup models
rank = 0  # Single process, so rank is 0
local_rank = 0  # Single process, so local_rank is 0
device = 'cpu'  # Use 'cpu' as you don't have a GPU

vqvae = make_vqvae(hps, device)
print_once(f"Parameters VQVAE:{count_parameters(vqvae)}")
if hps.prior:
    prior = make_prior(hps, vqvae, device)
    print_once(f"Parameters Prior:{count_parameters(prior)}")
    model = prior
else:
    model = vqvae

# Setup opt, ema and distributed_model.
opt, shd, scalar = get_optimizer(model, hps)
ema = get_ema(model, hps)
distributed_model = get_ddp(model, hps)

logger, metrics = init_logging(hps, local_rank, rank)
logger.iters = model.step

# Run training, eval, sample
for epoch in range(hps.curr_epoch, hps.epochs):
    metrics.reset()
    data_processor.set_epoch(epoch)
    if hps.train:
        train_metrics = train(distributed_model, model, opt, shd, scalar, ema, logger, metrics, data_processor, hps)
        train_metrics['epoch'] = epoch
        if rank == 0:
            print('Train',' '.join([f'{key}: {val:0.4f}' for key,val in train_metrics.items()]))
        dist.barrier()

    if hps.test:
        if ema: ema.swap()
        test_metrics = evaluate(distributed_model, model, logger, metrics, data_processor, hps)
        test_metrics['epoch'] = epoch
        if rank == 0:
            print('Ema',' '.join([f'{key}: {val:0.4f}' for key,val in test_metrics.items()]))
        dist.barrier()
        if ema: ema.swap()
    dist.barrier()