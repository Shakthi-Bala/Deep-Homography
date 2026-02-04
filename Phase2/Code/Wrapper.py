#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


import numpy as np
import cv2
import os 
import glob
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import  torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class HomographyDataset(Dataset):

    def __init__(self, root_dir, crop_size=128, boundary_limit=32, transformation=None):
        super().__init__()

        self.root_dir=root_dir
        self.crop_size=crop_size
        self.transformation=transformation
        self.image_path=glob.glob(os.path.join(root_dir,"*"))
        self.boundary_limit=boundary_limit

    
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,idx):

        img_path=self.image_path[idx]

        image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        #safety check (for image sizee)

        if image is None or image.shape[1]<self.crop_size+ 2*self.boundary_limit or image.shape[0]<self.crop_size+ 2*self.boundary_limit:
            return self.__getitem__(np.random.randint(0,len(self)))

        crop_height, crop_width = self.crop_size, self.crop_size
        patch_a, patch_b, ground_truth_H4p=self.get_random_crop_and_pertube(image,crop_height, crop_width)

        #Normalise Patch
        patch_a = patch_a.astype(np.float32) /255.0
        patch_b = patch_b.astype(np.float32) /255.0

        # Stack patch -> 2,128,128
        stacked_images=np.stack([patch_a ,patch_b],axis=0)

        #Normalise Delta H4p
        ground_truth_H4p= ground_truth_H4p/(self.boundary_limit-1)

        # Flatten H4p

        label=ground_truth_H4p.flatten()

        # convert to torch tensor 
        sample={
            "image":torch.from_numpy(stacked_images),
            "label":torch.from_numpy(label)
        }

        if self.transformation:
            sample=self.transformation(sample)

        return sample 


        


        

    def get_random_crop_and_pertube(self,image, crop_height=128, crop_width=128):
        """Get random crop of image and pertube"""
        if image.shape[1]>192 and image.shape[0]>192:
            boundary_limit=32
        
        elif image.shape[1]>160 and image.shape[0]>160:
            boundary_limit=16

        else:
            print("image to small, skip image")
            return None,None, None
        
        max_x= image.shape[1] - crop_width - boundary_limit
        max_y= image.shape[0] - crop_height - boundary_limit

        x= np.random.randint(boundary_limit, max_x)
        y= np.random.randint(boundary_limit, max_y)

        #Patch A 
        crop_image=image[y:y+crop_height, x:x+crop_width]
        crop_height_half=(y+crop_height)/2
        crop_width_half=(x+crop_width)/2
        
        # Top left top right, and bottom right, bottom left crop image positions
        # 4 corners list
        corner_positions=np.array([[x,y],[x+crop_width,y],[x+crop_width,y+crop_height]
                                ,[x,y+crop_height]],dtype = (np.float32))
        
        h,w=crop_image.shape
        
        pertube_range=boundary_limit-1 # avoiding 50 , due to boundary issue
        pertubation=np.random.randint(-pertube_range,pertube_range,(4,2)).astype(np.float32)

        new_pertube_corners=corner_positions+pertubation

        H=cv2.getPerspectiveTransform(corner_positions,new_pertube_corners)

        H_inv=np.linalg.inv(H)

        pertube_image=cv2.warpPerspective(image,H,(image.shape[1],image.shape[0]),flags=cv2.WARP_INVERSE_MAP)

        #Patch B 

        
        crop_and_pertube_result=pertube_image[y:y+crop_height, x:x+crop_width]


        return crop_image, crop_and_pertube_result, pertubation

class HomographyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)

        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)

        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)

        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            

        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*16*16, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024,8)
        )

    def forward(self,x):
        out =self.layer1(x)
        out =self.layer2(out)
        out =self.layer3(out)
        out =self.layer4(out)
        out = out.contiguous().view(x.size(0), -1)
        out = self.fc(out)
        return out 
        

class ModelTrainer:
    def __init__(self,model,train_loader, val_loader, lr=0.005,epochs=50, ):
        self.model=model.to(device)
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.lr=lr
        self.epochs=epochs
        self.criterion=nn.MSELoss()

        self.optimizer=optim.SGD(self.model.parameters(),self.lr,momentum=0.9)

        self.best_val_loss = float('inf')
        
    def train(self):
        print("Start Training ")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss=0.0

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")

            for batch in loop:
                images= batch["image"].to(device)
                labels= batch["label"].to(device)


                output=self.model(images)

                loss=self.criterion(output, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss+=loss.item()
                loop.set_postfix(loss=loss.item())

            
            # end of epoch stats
            avg_train_loss=running_loss/len(self.train_loader)

            avg_val_loss = self.validate()

            # Added the missing '}' after the first variable
            print(f"Epoch [{epoch+1}/{self.epochs}], Average Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model("best_model.pth")
                print("New best model saved!")

    def validate(self):
   
        self.model.eval() 
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_val_loss += loss.item()
        
        return running_val_loss / len(self.val_loader)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")







def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    train_dir="Deep-Homography/Phase2/Data/Train"
    val_dir="Deep-Homography/Phase2/Data/Val"
    dataset=HomographyDataset(train_dir, crop_size=128, boundary_limit=32, transformation=None)
    val_dataset=HomographyDataset(val_dir, crop_size=128, boundary_limit=32, transformation=None)

    train_loader=DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader=DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

            
    model=HomographyNet()

    trainer=ModelTrainer(model,train_loader=train_loader,val_loader=val_loader,lr=0.005, epochs=50 )

    trainer.train()

    trainer.save_model("final_homography_net.pth")


    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
