#-*- coding:utf-8 -*-
from diffusion_model.trainer import GaussianDiffusion, num_to_groups
from diffusion_model.unet import create_model
from torchvision import utils
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
import argparse
import glob
import cv2
import os
import numpy as np
from imgaug import augmenters as iaa
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import uuid
from torchvision.transforms import Compose, Lambda
from PIL import Image
import numpy as np
import torch
import os 
import glob
from dataset import SCDDPairImageGenerator
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import wandb

# +
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_path', type=str)
parser.add_argument('-w', '--weightfile', type=str)
parser.add_argument('-div', '--device', type=str, default='cuda')
parser.add_argument('--input_size', type=int, default=512)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--timesteps', type=int, default=1000)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('-c', '--cell_type', type=str, default= 'mono', help="mono, multi, mono halfcut, multi halfcut, dogbone")
parser.add_argument('-cat', '--category', type=str, default="crack", help="crack, gridline, (all defects)")

args = parser.parse_args()

wandb.init(
    name = "inference_mono_resume_all161_1",
    project = "pv-diffusion",
    entity = "anonym",
    config={
        "input_size" : args.input_size,
        "batchsize" : args.batchsize,
        "weightfile" : args.weightfile,
        "num_channels" : args.num_channels,
        "num_res_blocks" : args.num_res_blocks,
        "num_samples" : args.num_samples,
        "in_channels": 6,
        "out_channels" : 3,
        "device" : "cuda",
        "timesteps" : args.timesteps,
    }
)

export_folder = os.path.join(f"exports", wandb.run.name)
wandb.config.export_folder = export_folder

input_transform = Compose([
    ToPILImage(),
    Resize(wandb.config.input_size),
    ToTensor(),
    Lambda(lambda t: (t * 2) - 1)
])


# Initialize the dataset generator
dataset = SCDDPairImageGenerator(
    dataset_path=args.dataset_path,
    input_size=args.input_size,
    cell_type=args.cell_type,
    split="test",
    transform=input_transform,
    remove_padding=True,
)

# Extract binary masks
dataset.extract_masks()

inputfolder = dataset.revised_binary_mask_path
    
mask_list = sorted(glob.glob(f"{inputfolder}/*.jpg") + glob.glob(f"{inputfolder}/*.png"))
print("Total input masks: ", len(mask_list))

model = create_model(wandb.config.input_size, wandb.config.num_channels, wandb.config.num_res_blocks, in_channels=wandb.config.in_channels, out_channels=wandb.config.out_channels).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = wandb.config.input_size,
    timesteps = wandb.config.timesteps,   # number of steps
    loss_type = 'l2'    # L1 or L2
).cuda()

# +
weight = torch.load(wandb.config.weightfile)
diffusion.load_state_dict(weight['ema'])
print("Model Loaded!")

img_dir = export_folder + "/image"   
msk_dir = export_folder + "/mask"   
os.makedirs(img_dir, exist_ok=True)
os.makedirs(msk_dir, exist_ok=True)
# -

for k, inputfile in enumerate(mask_list):
    left = len(mask_list) - (k+1)
    print("MASKS LEFT: ", left)
    batches = num_to_groups(wandb.config.num_samples, wandb.config.batchsize)
    img = Image.open(inputfile)
    img = img.resize((wandb.config.input_size, wandb.config.input_size))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)
    input_tensor = input_transform(img)
    input_tensor = input_tensor.unsqueeze(0)
    msk_name = inputfile.split('/')[-1]
    
    steps = len(batches)
    sample_count = 0
    
    print(f"All Step: {steps}")
    counter = 0
    
    for i, bsize in enumerate(batches):
        print(f"Step [{i+1}/{steps}]")
        condition_tensors, counted_samples = [], []
        for b in range(bsize):
            condition_tensors.append(input_tensor)
            counted_samples.append(sample_count)
            sample_count += 1

        condition_tensors = torch.cat(condition_tensors, 0).cuda()
        imgs_list = list(map(lambda n: diffusion.sample(batch_size=n, condition_tensors=condition_tensors), [bsize]))
        # imgs_list = list(map(lambda n: diffusion.sample_from_image(init_image_path=args.init_image_path, condition_tensors=condition_tensors), [bsize]))
        

        # Iterate over each batch and each image in the batch
        for batch_idx, imgs in enumerate(imgs_list):
            imgs = (imgs + 1) * 0.5  # Normalize the images
            
            for img_idx, img in enumerate(imgs):
                counter = counter + 1
                # Generate a unique filename for each image
                filename = os.path.join(img_dir, f'{counter}-{msk_name}')
                utils.save_image(img, filename)
                # post-process the generated image to RGBA-like
                img_grayscale = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                # Normalize to [0,1] float and convert to tensor
                img_tensor = torch.tensor(img_grayscale, dtype=torch.float32) / 255.0  # shape: (H, W)
                # Expand to 3 channels (RGB)
                fake_rgb = img_tensor.unsqueeze(0).repeat(3, 1, 1)
                # Add alpha channel (1 = 255 = fully opaque)
                alpha_channel = torch.ones_like(img_tensor).unsqueeze(0)  # shape: (1, H, W)
                fake_rgba = torch.cat([fake_rgb, alpha_channel], dim=0)  # shape: (4, H, W)
                    
                # Save the image
                utils.save_image(fake_rgba, filename)
                wandb.log({
                    "Generated Image": wandb.Image(filename)
                })
                # Generate a unique filename for each image
                filename = os.path.join(msk_dir, f'{counter}-{msk_name}')
                utils.save_image(condition_tensors[0], filename)
                wandb.log({
                    "Condition Mask": wandb.Image(filename)
                })
        print("Done!")
