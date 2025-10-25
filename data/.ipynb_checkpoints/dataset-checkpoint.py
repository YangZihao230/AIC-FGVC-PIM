
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import copy
import torch
import warnings
from PIL import ImageFile
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .randaug import RandAugment


def build_loader(args):
    train_set, train_loader, val_set, val_loader = None, None, None, None

    # 1. 加载完整的训练集（包含待划分的所有样本）
    if args.train_root is not None:
        full_train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)

        # 2. 判断是否需要随机划分验证集
        if args.val_root is not None:
            # 情况1：有独立的val文件夹，直接加载
            train_set = full_train_set
            val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        else:
            # 情况2：无独立val文件夹，从训练集中随机划分（默认20%为验证集，可在yaml中加val_split参数）
            val_size = int(len(full_train_set) * args.val_split)  # args.val_split需在yaml中定义，如0.2
            train_size = len(full_train_set) - val_size
            # 用random_split划分，固定种子确保每次划分一致
            train_set, val_set = torch.utils.data.random_split(
                full_train_set, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # 固定种子，可复现
            )
            # 关键：验证集需用“测试/验证模式”的增强（无随机裁剪、翻转）
            val_set.dataset.istrain = False  # 把验证集的增强模式改成istrain=False
            val_set.dataset.transforms = transforms.Compose([  # 重新赋值验证集的增强逻辑
                transforms.Resize((510, 510), Image.BILINEAR),
                transforms.CenterCrop((args.data_size, args.data_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # 3. 构建DataLoader
    if train_set is not None:
        train_loader = torch.utils.data.DataLoader(
            train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size
        )
    if val_set is not None:
        val_loader = torch.utils.data.DataLoader(
            val_set, num_workers=1, shuffle=False, batch_size=args.batch_size  # 验证集不shuffle
        )

    return train_loader, val_loader

def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None

# 定义健壮加载器（放在ImageDataset类外）
_CORRUPTED_WARNED = set()
def robust_pil_loader(path):
    try:
        with Image.open(path) as img:
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.load()
            return img.copy()
    except Exception as e:
        if path not in _CORRUPTED_WARNED:
            print(f"Warning: Skipping corrupted image {path}. Error: {e}")
            _CORRUPTED_WARNED.add(path)
        # 返回灰色占位图（尺寸与data_size一致）
        return Image.new('RGB', (data_size, data_size), color=(128, 128, 128))

class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)

    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort()  # 保证类别ID稳定
        print("[dataset] class number:", len(folders))  # 应输出400
        for class_id, folder in enumerate(folders):
            # 用os.path.join拼接文件夹路径，自动处理斜杠
            folder_path = os.path.join(root, folder)
            # 跳过非文件夹（避免误将文件当作类别文件夹）
            if not os.path.isdir(folder_path):
                continue
            files = os.listdir(folder_path)
            for file in files:
                # 拼接图像完整路径
                data_path = os.path.join(folder_path, file)
                data_infos.append({"path": data_path, "label": class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]

        # 用健壮加载器读取图像（替换cv2.imread）
        img = robust_pil_loader(image_path)  # 自动处理损坏图像，返回RGB格式PIL图像
        img = self.transforms(img)  # 直接应用增强（无需BGR→RGB转换，PIL默认RGB）

        if self.return_index:
            return index, img, label
        return img, label