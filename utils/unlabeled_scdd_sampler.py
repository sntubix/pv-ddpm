import os
import shutil
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import random

def check_sample_info(data_info_path, sampled_image):
  if os.path.exists(data_info_path):
    data_info = pd.read_csv(data_info_path)
  else:
    data_info = pd.DataFrame(columns=['Path', 'Sampled images'])
    data_info.to_csv(data_info_path, index=False)  
  
  if data_info.empty or sampled_image not in data_info['Sampled images'].values:
    is_sampled = False
  else:
      is_sampled = True
  return is_sampled


def update_info(data_info_path, folder, ribbon, sampled_image):
  data_info = pd.read_csv(data_info_path)
  new_row = pd.DataFrame([{'Path': f"{folder}/{ribbon}", 'Sampled images': sampled_image}])
  data_info = pd.concat([data_info, new_row], ignore_index=True)
  data_info.to_csv(data_info_path, index=False)
  

def image_sampler(dataset_path, folder, ribbon, samples, data_info_path):
  if folder is None:
    folders = os.listdir(dataset_path)
  else:
    folders = [folder]
  for folder in folders:
    if ribbon is None:
      folders_path = os.path.join(dataset_path, folder)
      if os.path.exists(folders_path):
        ribbons = os.listdir(os.path.join(dataset_path, folder))
      else:
        raise Exception(f"Ribbons not found in the following path: {folders_path}")
    else:
      ribbons = [ribbon]
    for ribbon in ribbons:
      images_path = os.path.join(dataset_path, folder, ribbon)
      if os.path.exists(images_path):
        images = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpg', '.png', '.jpeg'))])
      else:
        raise Exception(f"Images not found in the following path: {images_path}")
      
      available_images = [img for img in images if not check_sample_info(data_info_path, img)]
      if len(available_images) < samples:
          raise ValueError("Not enough unique images to sample")
      
      sampled_images = random.sample(available_images, samples)
      for sample_image in sampled_images:
        update_info(data_info_path, folder, ribbon, sample_image)
        sampled_images_path = os.path.join(images_path, "sampled_images")
        os.makedirs(sampled_images_path, exist_ok=True)
        shutil.copy(os.path.join(images_path, sample_image), os.path.join(sampled_images_path, sample_image))
       

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset_path', type=str)
  parser.add_argument('-f', '--folder', type=str, default="multi", help="Specify the folder to sample images from: mono, multi, multi_half_cells")
  parser.add_argument('-r', '--ribbon', type=str, default="ribbons_4", help="Specify the ribbons to sample images from")
  parser.add_argument('-s', '--samples', type=int, default=20, help="Specify the number of samples to be selected")
  args = parser.parse_args()

  data_info_path = os.path.join(args.dataset_path, "sampled_data_info.csv")
  statistics_info_path = os.path.join(args.dataset_path, "sampled_statistics_info.csv")
  image_sampler(args.dataset_path, args.folder, args.ribbon, args.samples, data_info_path)
  shutil.copy(data_info_path, "./sampled_data_info.csv")
      
