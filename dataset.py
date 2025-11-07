#-*- coding:utf-8 -*-
# +
import os
import re
import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import cv2
import shutil

class JPGPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            transform=None
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.transform = transform

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*.jpg')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*.jpg')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            pairs.append((input_file, target_file))
        return pairs

    def read_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((256, 256))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        return img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file)
            if self.transform:
                input_img = self.transform(input_img)
                input_tensors.append(input_img)
        return torch.stack(input_tensors).cuda()
    
    
    def sample_pairs(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        samples = [self.__getitem__(index) for index in indexes]
        input_imgs, target_imgs = zip(*[(sample['input'], sample['target']) for sample in samples])
        return torch.stack(input_imgs), torch.stack(target_imgs)

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file)
        target_img = self.read_image(target_file)

        if self.transform:
            combined_img = np.concatenate((input_img, target_img), axis=2)
            combined_img = self.transform(combined_img)
            input_img, target_img = torch.split(combined_img, 3, dim=0)

        return {'input': input_img, 'target': target_img}

class SCDDPairImageGenerator(Dataset):
    def __init__(self,
            dataset_path: str,
            input_size: int,
            cell_type: str,
            split: str,
            transform=None,
            remove_padding: bool = True,
        ):
        
        self.dataset_path = dataset_path
        self.dataset_path_elements, self.class_label_df, self.class_label_dict = self.load_dataset_info()
        self.input_size = input_size
        self.transform = transform
        self.cell_type = cell_type
        self.split = split
        self.remove_padding = remove_padding
        self.images_folder_path, self.masks_folder_path, self.non_augmented_data_list = self.get_dataset_paths_and_files()
        if self.remove_padding:
            self.input_data_list, self.images_folder_path, self.masks_folder_path = self.remove_padding_resize_folder()
        if self.cell_type:
            self.revised_binary_mask_path = os.path.join(self.masks_folder_path, "revised_binary_masks", self.cell_type)
            self.cell_type_data_list = self.separate_cell_type(self.cell_type) 
            self.input_data_list = self.cell_type_data_list
        else:
            self.revised_binary_mask_path = os.path.join(self.masks_folder_path, "revised_binary_masks", "all")
            # self.input_data_list = self.non_augmented_data_list
        if os.path.exists(self.revised_binary_mask_path):
            shutil.rmtree(self.revised_binary_mask_path)
        # Create the directory
        os.makedirs(self.revised_binary_mask_path, exist_ok=True)
        
        self.extract_masks()
        self.pair_files = self.pair_file()       

    def load_dataset_info(self):
        """
        Load dataset information from dataset path including the classes and their related labels
        """
        dataset_path_elements = os.listdir(self.dataset_path)

        class_label_path = os.path.join(self.dataset_path, str([element for element 
                                                    in dataset_path_elements 
                                                    if "csv" in element and "List" in element][0]))
        if class_label_path is None:
            raise Exception("CSV file describing the classes not found in the following path: {}".format(self.dataset_path))
        
        class_label_df = pd.read_csv(class_label_path)
        class_label_dict = dict(zip(class_label_df['Desc'], 
                                class_label_df['Label']))
        return dataset_path_elements, class_label_df, class_label_dict
    
    def get_dataset_paths_and_files(self):
        """
        Get the non-augmented fie names from the dataset
        """
        images_folder_name = [element for element in self.dataset_path_elements if 'image' in element and self.split in element][0]        
        if not images_folder_name:
            raise ValueError("Image folder for training not found in the following path: {}".format(self.dataset_path))
        images_folder_path = os.path.join(self.dataset_path, images_folder_name)
        
        masks_folder_name = [element for element in self.dataset_path_elements if 'mask' in element and self.split in element][0]
        if not masks_folder_name:
            raise ValueError("Mask folder for training not found in the following path: {}".format(self.dataset_path))
        masks_folder_path = os.path.join(self.dataset_path, masks_folder_name)
        
        non_augmented_data_list = [
            f for f in os.listdir(images_folder_path)
            if f.endswith(('.jpg', '.png', '.jpeg')) and not any(keyword in f for keyword in ['rotate', 'mirror', 'flip'])
        ]    
        
        return images_folder_path, masks_folder_path, non_augmented_data_list
    
    def separate_cell_type(self, cell_type: str = None):
        """
        Separate data based on the cell type
        """
        cell_type = cell_type if cell_type else self.cell_type  
        cell_type_description = [desc for desc in self.class_label_dict.keys() if "sp" in desc and cell_type in desc][0]           
        cell_type_label = self.class_label_dict[cell_type_description]
        
        if cell_type_label is None:
            raise ValueError("Cell type not found in the class label dictionary.")
        
        cell_type_data_list = []
        
        for mask in self.input_data_list:
            mask_path = os.path.join(self.masks_folder_path, mask)

            if not os.path.exists(mask_path):   
                raise ValueError(f"Mask file not found: {mask_path}")
            
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                
            if cell_type_label in mask_img:
                cell_type_data_list.append(mask)
                
        return cell_type_data_list
    
    def remove_padding_resize_folder(self):
        """
        Remove padding from the images and masks
        """
        padding_label = self.class_label_dict["padding"]
        
        no_padding_images_folder_path = os.path.join(self.images_folder_path, "removed_padding")
        if os.path.exists(no_padding_images_folder_path):
            shutil.rmtree(no_padding_images_folder_path)
        shutil.copytree(self.images_folder_path, no_padding_images_folder_path)
        self.images_folder_path = no_padding_images_folder_path
        
        no_padding_masks_folder_path = os.path.join(self.masks_folder_path, "removed_padding")
        if os.path.exists(no_padding_masks_folder_path):
            shutil.rmtree(no_padding_masks_folder_path)
        shutil.copytree(self.masks_folder_path, no_padding_masks_folder_path)
        self.masks_folder_path = no_padding_masks_folder_path
        
        for mask in self.non_augmented_data_list:
            mask_path = os.path.join(self.masks_folder_path, mask)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if np.any(mask_img == padding_label):
                el_image_path = os.path.join(self.images_folder_path, mask)
                el_image = cv2.imread(el_image_path, cv2.IMREAD_UNCHANGED)
                
                non_padding_area = (mask_img != padding_label).astype(np.uint8)
                coords = cv2.findNonZero(non_padding_area)
                x, y, w, h = cv2.boundingRect(coords)
                cropped_mask = mask_img[y:y+h, x:x+w]
                scale = self.input_size / min(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                while new_w < self.input_size or new_h < self.input_size:
                    scale = scale * 1.1
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                # Resize the mask to the new dimensions
                resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                #center crop
                start_x = (new_w - self.input_size) // 2
                start_y = (new_h - self.input_size) // 2
                final_cropped_mask = resized_mask[start_y:start_y+self.input_size, start_x:start_x+self.input_size]
                # final_cropped_mask_path = os.path.join(removed_padding_path, f"el_masks_{self.split}", f"{mask.split('.')[0]}_no_padding.png")
                final_cropped_mask_path = os.path.join(f"{mask_path.split('.')[0]}_no_padding.png")
                os.makedirs(os.path.dirname(final_cropped_mask_path), exist_ok=True)
                cv2.imwrite(final_cropped_mask_path, final_cropped_mask)
                
                cropped_image = el_image[y:y+h, x:x+w]
                resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                final_cropped_image = resized_image[start_y:start_y+self.input_size, start_x:start_x+self.input_size]
                final_cropped_image_path = os.path.join(f"{el_image_path.split('.')[0]}_no_padding.png")
                os.makedirs(os.path.dirname(final_cropped_image_path), exist_ok=True)
                cv2.imwrite(final_cropped_image_path, final_cropped_image)
                os.remove(mask_path)
                os.remove(el_image_path)
                
                self.input_data_list = [ f for f in os.listdir(self.images_folder_path)
                if f.endswith(('.jpg', '.png', '.jpeg'))]    
                
        return self.input_data_list, self.images_folder_path, self.masks_folder_path
                
            
    def remove_padding_resize_file(self, file, mask_img):
        """
        Remove padding from the images and masks
        """
        padding_label = self.class_label_dict["padding"]
        
        no_padding_images_folder_path = os.path.join(self.images_folder_path, "removed_padding")
        os.makedirs(no_padding_images_folder_path, exist_ok=True)
        shutil.copytree(self.images_folder_path, no_padding_images_folder_path)
        self.images_folder_path = no_padding_images_folder_path
        
        no_padding_masks_folder_path = os.path.join(self.masks_folder_path, "removed_padding")
        os.makedirs(no_padding_masks_folder_path, exist_ok=True)
        shutil.copytree(self.masks_folder_path, no_padding_masks_folder_path)
        self.masks_folder_path = no_padding_masks_folder_path
        

        if np.any(mask_img == padding_label):
            el_image_path = os.path.join(self.images_folder_path, file)
            el_image = cv2.imread(el_image_path, cv2.IMREAD_UNCHANGED)
            
            non_padding_area = (mask_img != padding_label).astype(np.uint8)
            coords = cv2.findNonZero(non_padding_area)
            x, y, w, h = cv2.boundingRect(coords)
            cropped_mask = mask_img[y:y+h, x:x+w]
            scale = self.input_size / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            while new_w < self.input_size or new_h < self.input_size:
                scale = scale * 1.1
                new_w = int(w * scale)
                new_h = int(h * scale)
            # Resize the mask to the new dimensions
            resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            #center crop
            start_x = (new_w - self.input_size) // 2
            start_y = (new_h - self.input_size) // 2
            final_cropped_mask = resized_mask[start_y:start_y+self.input_size, start_x:start_x+self.input_size]
            final_cropped_mask_path = os.path.join(self.masks_folder_path, f"{file.split('.')[0]}_no_padding.png")
            os.makedirs(os.path.dirname(final_cropped_mask_path), exist_ok=True)
            cv2.imwrite(final_cropped_mask_path, final_cropped_mask)
            
            cropped_image = el_image[y:y+h, x:x+w]
            resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            final_cropped_image = resized_image[start_y:start_y+self.input_size, start_x:start_x+self.input_size]
            final_cropped_image_path = os.path.join(self.images_folder_path, f"{file.split('.')[0]}_no_padding.png")
            os.makedirs(os.path.dirname(final_cropped_image_path), exist_ok=True)
            cv2.imwrite(final_cropped_image_path, final_cropped_image)
            
            
            
    def extract_masks(self):
        """
        Pair each image with a background mask or combination of features and defect masks.
        """
        
        for file in self.input_data_list:
            image_path = os.path.join(self.images_folder_path, file)
            mask_path = os.path.join(self.masks_folder_path, file)
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            if not os.path.exists(mask_path):
                raise ValueError(f"Mask file not found: {mask_path}")
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if not os.path.exists(self.revised_binary_mask_path):
                os.makedirs(self.revised_binary_mask_path, exist_ok=True)
    
            
            feature_desc_list = self.class_label_df[self.class_label_df["Class"].str.lower() == "feature"]["Desc"].tolist()
            feature_desc_list.remove("padding") 
            feature_desc_list.remove("text") 
            defect_desc_list = self.class_label_df[self.class_label_df["Class"].str.lower() == "defect"]["Desc"].tolist()
            
            available_features = []
            for feature_desc in feature_desc_list:
                # bckgnd pairing
                if feature_desc == "bckgnd":
                    bckgrnd_label = self.class_label_dict[feature_desc]
                    bckgnd_binary_mask = np.where(mask_img == bckgrnd_label, 255, 0).astype(np.uint8)
                    bckgnd_binary_mask = cv2.resize(bckgnd_binary_mask, (self.input_size, self.input_size))
                    bckgnd_binary_mask_path = os.path.join(self.revised_binary_mask_path, f'{file.split(".")[0]}_bckgnd.png')
                    cv2.imwrite(bckgnd_binary_mask_path, bckgnd_binary_mask)
                    # pairs.append((bckgnd_binary_mask_path, image_path))
                else:
                    if self.class_label_dict[feature_desc] in mask_img:
                        available_features.append(feature_desc)
            for defect_desc in defect_desc_list:
                defect_label = self.class_label_dict[defect_desc]
                defect_binary_mask = np.zeros_like(mask_img)
                if defect_label in mask_img:
                    for feature in available_features:
                        # defect_binary_mask = np.where(mask_img == self.class_label_dict[feature], 255, 0).astype(np.uint8)
                        defect_binary_mask[mask_img == self.class_label_dict[feature]] = 255
                    # defect_binary_mask = np.where((mask_img == defect_label), 255, 0).astype(np.uint8)
                    defect_binary_mask[mask_img == defect_label] = 255
                    defect_binary_mask = cv2.resize(defect_binary_mask, (self.input_size, self.input_size))
                    defect_binary_mask_path = os.path.join(self.revised_binary_mask_path, f'{file.split(".")[0]}_{defect_desc}.png')
                    cv2.imwrite(defect_binary_mask_path, defect_binary_mask)                   

    
    def pair_file(self):
        pairs = []
        masks = [mask for mask in os.listdir(self.revised_binary_mask_path) if mask.endswith(('.jpg', '.png', '.jpeg'))]
        images = [image for image in os.listdir(self.images_folder_path) if image.endswith(('.jpg', '.png', '.jpeg'))]
        
        for file in self.input_data_list:
            if file in images:
                matching_masks = [mask for mask in masks if mask.startswith(file.split(".")[0])]
                if not matching_masks:
                    raise ValueError(f"No matching mask found for: {file}")
                image_path = os.path.join(self.images_folder_path, file)
                
                for mask in matching_masks:
                    mask_path = os.path.join(self.revised_binary_mask_path, mask)
                    pairs.append((mask_path, image_path))
        print(f"Number of pairs: {len(pairs)}")
        # print(f"Pairs: {pairs}")
        
        return pairs
            
        

    def read_image(self, file_path):
        img = Image.open(file_path)
        img = img.resize((self.input_size, self.input_size))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        return img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file)
            if self.transform:
                input_img = self.transform(input_img)
                input_tensors.append(input_img)
        return torch.stack(input_tensors).cuda()
    
    
    def sample_pairs(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        samples = [self.__getitem__(index) for index in indexes]
        input_imgs, target_imgs = zip(*[(sample['input'], sample['target']) for sample in samples])
        return torch.stack(input_imgs), torch.stack(target_imgs)

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file)
        target_img = self.read_image(target_file)

        if self.transform:
            combined_img = np.concatenate((input_img, target_img), axis=2)
            combined_img = self.transform(combined_img)
            input_img, target_img = torch.split(combined_img, 3, dim=0)

        return {'input': input_img, 'target': target_img}

if __name__ == "__main__":
    dataset_path = input("Enter the dataset path: ")
    dataset = SCDDPairImageGenerator(
        dataset_path=dataset_path,
        input_size=512,
        cell_type=None,
        split="test",
        )
    dataset.remove_padding_resize()
        
