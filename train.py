import torchvision.transforms as transforms
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import SCDDPairImageGenerator
import argparse
import torch
import os
import pandas as pd
import wandb
import torch.onnx


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# -

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_path', type=str)
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--timesteps', type=int, default=1000)
parser.add_argument('-r', '--resume_weight', type=str)
parser.add_argument('-c', '--cell_type', type=str, default= "mono halfcut", help="mono, multi, mono halfcut, multi halfcut, dogbone")
parser.add_argument('-cat', '--category', type=str, default="crack", help="crack, gridline, (all defects)")
parser.add_argument('-p', '--early_stopping_patience', type=int, default=10)

args = parser.parse_args()

dataset_path = args.dataset_path
input_size = args.input_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
batchsize = args.batchsize
epochs = args.epochs
timesteps = args.timesteps
resume_weight = args.resume_weight
cell_type = args.cell_type
category = args.category
early_stopping_patience = args.early_stopping_patience

# split = args.split
in_channels = 6
out_channels = 3

wandb.init(
    name = "train_monohalfcut_resume161_nomono",
    project = "pv-diffusion",
    entity = "anonym",
    config={
        "dataset_path" : args.dataset_path,
        "input_size" : args.input_size,
        "num_channels" : args.num_channels,
        "num_res_blocks" : args.num_res_blocks,
        "batchsize" : args.batchsize,
        "epochs" : args.epochs,
        "timesteps" : args.timesteps,
        "resume_weight ": args.resume_weight,
        "cell_type" : args.cell_type,
        "in_channels": in_channels,
        "out_channels" : out_channels,
        "loss_type" : 'l2',
        "early_stopping_patience" : args.early_stopping_patience,
    }    
)

results_folder = os.path.join(f"./results/", wandb.run.name)
wandb.config.results_folder = results_folder


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=180), 
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda t: (t * 2) - 1),
])


dataset = SCDDPairImageGenerator(
    dataset_path=dataset_path,
    input_size=input_size,
    transform=transform,
    cell_type=cell_type,
    split="train",
    remove_padding=True,
    )

# Create validation dataset
val_dataset = SCDDPairImageGenerator(
    dataset_path=dataset_path,
    input_size=input_size,
    transform=transform,
    cell_type=cell_type,
    split="val",
    remove_padding= True,
)

# Log training image-mask pairs
df_pairs = pd.DataFrame(dataset.pair_files, columns=["image", "mask"])  
df_pairs_filenames = pd.DataFrame({
    'image_filename': df_pairs['image'].apply(lambda x: x.split('/')[-1]),
    'mask_filename': df_pairs['mask'].apply(lambda x: '/'.join(x.split('/')[-2:]))
})
wandb.log({"trainig_pairs":wandb.Table(dataframe=df_pairs_filenames)})

model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = wandb.config.loss_type   # L1 or L2 or perceptual
).cuda()


#if len(resume_weight) > 0:
if resume_weight is not None:
    weight = torch.load(resume_weight, map_location='cuda')
    diffusion.load_state_dict(weight['ema'])
    print("Model Loaded!")
# -

# Calculate total steps
total_steps = args.epochs * len(dataset) // args.batchsize



trainer = Trainer(
    diffusion,
    dataset,
    image_size = input_size,
    train_batch_size = args.batchsize,
    train_lr = 5e-4,
    train_num_steps = total_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = True,
    save_and_sample_every = wandb.config.epochs -1,
    results_folder = results_folder,
    val_dataset = val_dataset,
    # val_batch_size = args.batchsize,
    early_stopping_patience = early_stopping_patience,
)

training_info = trainer.train()

# Log losses
for step, loss in training_info['losses']:
    wandb.log({"loss": loss, "step": step})

# Log samples and models
for step, sample_path in training_info['samples']:
    wandb.log({
        "samples": wandb.Image(sample_path),
        "step": step
    })
    
# Log model artifacts
for step, model_path in training_info['models']:
    model_artifact = wandb.Artifact(
        f"model-step-{step}", 
        type="model",
        description=f"Model checkpoint at step {step}"
    )
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    
# Log validation losses
for step, val_loss in training_info['val_losses']:
    wandb.log({"val_loss": val_loss, "step": step})
    
# Log the best model to wandb
best_model_path = os.path.join(results_folder, "model-best.pt")
if os.path.exists(best_model_path):
    model_artifact = wandb.Artifact(
        "best_model", 
        type="model",
        description="Best model based on validation loss"
    )
    model_artifact.add_file(best_model_path)
    wandb.log_artifact(model_artifact)
    print(f"Best model logged to wandb: {best_model_path}")
else:
    print("Best model file not found. Ensure the Trainer class saves it correctly.")
    
    
