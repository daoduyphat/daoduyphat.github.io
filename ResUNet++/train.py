import os
import random
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd

#goi cac models
from models.unet import UNet
from models.unetpp import NestedUNet
from models.resattention_unetpp import ResAttentionUNetPP

#dataset
class FetalHeadDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=(256,256), normalize=True):
        self.image_dir = Path(image_dir) #thu muc ahh
        self.label_dir = Path(label_dir) #thu muc label
        #chi lay cai nao co anh va mask
        self.files = sorted([f for f in os.listdir(self.image_dir) if f in os.listdir(self.label_dir)])
        if len(self.files) == 0:
            raise RuntimeError(f"Khong khop giua {image_dir} va {label_dir}")
        self.img_size = img_size #resize anh va mask
        self.normalize = normalize #chuan hoa anh ve [0,1]
        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(img_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.image_dir / fname
        lbl_path = self.label_dir / fname

        image = Image.open(img_path).convert("L")
        label = Image.open(lbl_path).convert("L")

        image = self.resize(image)
        label = self.resize(label)

        image = self.to_tensor(image)
        if self.normalize:
            image = (image - 0.5) / 0.5

        label = self.to_tensor(label)
        label = (label > 0.5).float()

        return image.float(), label.float()

#metrics va loss
def dice_coef(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    denom = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (denom + smooth)
    return dice.mean().item()

def iou_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
        self.bce_weight = bce_weight

    def forward(self, preds, targets):
        # Nếu preds là list (deep supervision), tính trung bình loss
        if isinstance(preds, list):
            losses = [self._compute_loss(p, targets) for p in preds]
            return sum(losses) / len(losses)
        else:
            return self._compute_loss(preds, targets)

    def _compute_loss(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        preds_sigmoid = torch.sigmoid(preds)
        intersection = (preds_sigmoid * targets).sum(dim=(1,2,3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            preds_sigmoid.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + self.smooth
        )
        dice_loss = dice_loss.mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

#train va valid
def valid_model(model, val_loader, criterion, device):
    model.eval()
    val_loss=0.0; dices=[]; ious=[]
    with torch.no_grad():
        for imgs,masks in val_loader:
            imgs,masks=imgs.to(device),masks.to(device)
            preds=model(imgs)
            loss=criterion(preds,masks).item()
            val_loss+=loss

            # lấy output cuối cùng nếu là list
            if isinstance(preds, list):
                preds_eval = preds[-1]
            else:
                preds_eval = preds

            dices.append(dice_coef(preds_eval,masks))
            ious.append(iou_score(preds_eval,masks))
    return val_loss/len(val_loader), float(np.mean(dices)), float(np.mean(ious))

def train_model(model, train_loader, val_loader, device, epochs, lr, output_dir, model_name="model"):
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    criterion=WeightedBCEDiceLoss()
    best_dice=0.0
    history={"train_loss":[], "val_loss":[], "dice":[], "iou":[]}
    for epoch in range(1,epochs+1):
        model.train(); train_loss=0.0
        loop=tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{epochs}")
        for imgs,masks in loop:
            imgs,masks=imgs.to(device),masks.to(device)
            preds=model(imgs); loss=criterion(preds,masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss+=loss.item(); loop.set_postfix(loss=loss.item())
        avg_train_loss=train_loss/len(train_loader)
        val_loss,val_dice,val_iou=valid_model(model,val_loader,criterion,device)
        print(f"{model_name} Epoch {epoch}/{epochs} | Train Loss {avg_train_loss:.4f} | Val Loss {val_loss:.4f} | Dice {val_dice:.4f} | IoU {val_iou:.4f}")
        history["train_loss"].append(avg_train_loss); history["val_loss"].append(val_loss)
        history["dice"].append(val_dice); history["iou"].append(val_iou)
        if val_dice>best_dice:
            best_dice=val_dice
            torch.save(model.state_dict(), os.path.join(output_dir,f"{model_name}_best.pth"))
    return history

#bat dau chay
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed);
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--images",type=str,default="dataset/images")
    p.add_argument("--labels",type=str,default="dataset/labels")
    p.add_argument("--output",type=str,default="outputs")
    p.add_argument("--epochs",type=int,default=15)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--img-size",type=int,nargs=2,default=(256,256))
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()

if __name__=="__main__":
    args=parse_args(); set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    dataset=FetalHeadDataset(args.images,args.labels,img_size=tuple(args.img_size))
    all_files=dataset.files; split=int(0.8*len(all_files))
    train_files=all_files[:split]; val_files=all_files[split:]
    train_dataset=FetalHeadDataset(args.images,args.labels,img_size=tuple(args.img_size)); train_dataset.files=train_files
    val_dataset=FetalHeadDataset(args.images,args.labels,img_size=tuple(args.img_size)); val_dataset.files=val_files
    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    os.makedirs(args.output,exist_ok=True)

    histories={}

    #train unet
    unet=UNet(in_ch=1,out_ch=1)
    histories["unet"]=train_model(unet,train_loader,val_loader,device,args.epochs,args.lr,args.output,model_name="unet")

    #train unet++
    unetpp=NestedUNet(in_ch=1,out_ch=1)
    histories["unetpp"]=train_model(unetpp,train_loader,val_loader,device,args.epochs,args.lr,args.output,model_name="unetpp")

    #train mo hinh bai bao
    resatt_unetpp = ResAttentionUNetPP(num_classes=1, input_channels=1)
    histories["resatt_unetpp"]=train_model(resatt_unetpp,train_loader,val_loader,device,args.epochs,args.lr,args.output,model_name="resatt_unetpp")

    #luu so lieu vao file excel
    df=pd.DataFrame({"Epoch":list(range(1,args.epochs+1))})
    for model_name,h in histories.items():
        df[f"{model_name}_Train_Loss"]=h["train_loss"]
        df[f"{model_name}_Val_Loss"]=h["val_loss"]
        df[f"{model_name}_Dice"]=h["dice"]
        df[f"{model_name}_IoU"]=h["iou"]
    out_excel=os.path.join(args.output,"bieudokq.xlsx")
    df.to_excel(out_excel,index=False)
    print(f"Da luu vao {out_excel}")
    print(df.tail())

    #ve bieu do loss, dice, iou
    plt.figure(figsize=(18,5))
    for i,metric in enumerate(["Train_Loss","Val_Loss","Dice","IoU"]):
        plt.subplot(1,4,i+1)
        for model_name in histories.keys():
            plt.plot(df["Epoch"], df[f"{model_name}_{metric}"], label=model_name, marker="o")
        plt.xlabel("Epoch"); plt.title(metric)
        plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output,"all_metrics.png"), dpi=150)
    plt.close()
