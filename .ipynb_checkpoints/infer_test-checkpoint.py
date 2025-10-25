import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import warnings
import contextlib
from argparse import Namespace

# -------------------------- 1. 导入项目核心模块（需确保路径正确）--------------------------
# 若脚本不在项目根目录，需添加项目路径到Python环境（例如：sys.path.append("../")）
from models.builder import MODEL_GETTER
from utils.config_utils import load_yaml, get_args
from utils.costom_logger import timeLogger

# -------------------------- 2. 工具函数：健壮图像加载器（处理损坏/截断图像）--------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
ImageFile.LOAD_TRUNCATED_IMAGES = True
_CORRUPTED_WARNED = set()


def robust_pil_loader(path, data_size):
    """健壮的图像加载器：处理损坏图像，返回RGB格式PIL图像"""
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
        # 损坏图像返回灰色占位图（尺寸与模型输入一致）
        return Image.new('RGB', (data_size, data_size), color=(128, 128, 128))


# -------------------------- 3. 测试集Dataset（适配无标签单文件夹图像）--------------------------
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_root: str, data_size: int):
        self.test_root = test_root
        self.data_size = data_size
        # 测试集数据增强（与验证集一致，无随机操作，确保结果稳定）
        self.transform = transforms.Compose([
            transforms.Resize((510, 510), Image.BILINEAR),  # 与训练时的Resize一致
            transforms.CenterCrop((data_size, data_size)),  # 固定中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet归一化（与训练一致）
                                 std=[0.229, 0.224, 0.225])
        ])
        # 读取测试集所有图像的路径和文件名（"id"为文件名）
        self.image_infos = self._get_image_infos()

    def _get_image_infos(self):
        """遍历测试集文件夹，收集图像路径和文件名"""
        image_infos = []
        for filename in os.listdir(self.test_root):
            file_path = os.path.join(self.test_root, filename)
            if os.path.isfile(file_path):  # 只保留文件（跳过子文件夹）
                # "id"使用文件名（如"test_img_001.jpg"），后续直接存入CSV
                image_infos.append({"path": file_path, "id": filename})
        print(f"[Test Dataset] Total images loaded: {len(image_infos)}")
        return image_infos

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, index):
        info = self.image_infos[index]
        img_path = info["path"]
        img_id = info["id"]  # CSV的"id"列内容（文件名）

        # 加载图像并应用增强
        img = robust_pil_loader(img_path, self.data_size)
        img_tensor = self.transform(img)

        return img_tensor, img_id  # 返回：图像张量、图像ID（文件名）


# -------------------------- 4. 核心推理函数--------------------------
def load_best_model(args, device):
    """加载训练保存的best.pt模型"""
    # 1. 构建与训练一致的模型结构
    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner
    )
    # 2. 加载best.pt权重（路径：save_dir/backup/best.pt）
    best_model_path = os.path.join(args.save_dir, "backup", "best.pt")
#     best_model_path = os.path.join(args.save_dir, "backup", "last.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found! Path: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])  # 加载模型权重
    model.to(device)
    model.eval()  # 设为评估模式（关闭Dropout、BatchNorm固定）
    print(f"Successfully loaded best model from: {best_model_path}")
    return model


def get_class_names(train_root):
    """获取类别名列表（与训练时的class_id对应，确保预测类别名正确）"""
    # 训练时类别ID按文件夹排序生成，此处需保持一致
    class_names = sorted(os.listdir(train_root))
    print(f"[Class Info] Total classes: {len(class_names)}")
    return class_names


def infer_test_set(args, device, tlogger):
    """批量推理测试集，保存结果到CSV"""
    # 1. 加载测试集
    tlogger.print("Loading test dataset...")
    test_dataset = TestDataset(
        test_root=args.test_root,
        data_size=args.data_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,  # 从配置文件取批量大小（适配GPU显存）
        shuffle=False,  # 测试集无需打乱
        num_workers=args.num_workers,  # 从配置文件取线程数
        pin_memory=True  # 加速数据传输到GPU
    )

    # 2. 加载best模型和类别名
    model = load_best_model(args, device)
    class_names = get_class_names(args.train_root)  # 从训练集路径获取类别名（确保ID对应）

    # 3. 初始化AMP上下文（与训练一致，若未启用则用nullcontext）
    if args.use_amp:
        amp_context = torch.cuda.amp.autocast
    else:
        amp_context = contextlib.nullcontext

    # 4. 批量推理
    tlogger.print("Start inferring test set...")
    infer_results = []  # 存储最终结果：[{"id": "...", "class": "..."}]
    total_batches = len(test_loader)

    with torch.no_grad():  # 关闭梯度计算（节省显存+加速推理）
        for batch_idx, (img_tensors, img_ids) in enumerate(test_loader):
            # 数据移到设备（GPU/CPU）
            img_tensors = img_tensors.to(device)

            # 前向传播（获取模型输出）
            with amp_context():
                outs = model(img_tensors)

            # 提取最终预测结果（根据PIM模型结构，优先取comb_outs；若无则取ori_out）
            if "comb_outs" in outs:
                pred_logits = outs["comb_outs"]  # 组合器输出（训练时的主要预测）
            elif "ori_out" in outs:
                pred_logits = outs["ori_out"]  # 原始输出（无PIM模块时）
            else:
                raise KeyError("Model output has no 'comb_outs' or 'ori_out'! Check model structure.")

            # 计算预测类别ID（取logits最大值对应的索引）
            pred_class_ids = torch.argmax(pred_logits, dim=1).cpu().numpy()  # 转CPU+Numpy

            # 映射类别ID到类别名，组装结果
            for img_id, pred_id in zip(img_ids, pred_class_ids):
                pred_class_name = class_names[pred_id]  # ID→类别名（如0→"Brewer_Blackbird"）
                infer_results.append({
                    "id": img_id,  # 列1：图像ID（文件名）
                    "class": pred_class_name  # 列2：预测类别名
                })

            # 打印推理进度
            if (batch_idx + 1) % args.log_freq == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                tlogger.print(f"Infer Progress: {progress:.1f}% (Batch {batch_idx + 1}/{total_batches})")

    # 5. 保存结果到CSV（路径：save_dir/test_infer_results.csv）
    result_df = pd.DataFrame(infer_results)
    csv_save_path = os.path.join(args.save_dir, "test_infer_results_adamw25.csv")
    result_df.to_csv(csv_save_path, index=False, encoding="utf-8")  # 不保存索引，UTF-8编码适配中文
    tlogger.print(f"Infer completed! Results saved to: {csv_save_path}")


# -------------------------- 5. 脚本入口（解析配置+启动推理）--------------------------
if __name__ == "__main__":
    # 初始化时间日志器（记录推理耗时）
    tlogger = timeLogger()
    tlogger.print("=" * 50)
    tlogger.print("Starting Test Set Inference Script")
    tlogger.print("=" * 50)

    # 1. 解析命令行参数（获取配置文件路径）
    args = get_args()
    # 断言：必须提供配置文件（yaml格式）
    assert args.c != "", "Please provide config file via '-c your_config.yaml'!"

    # 2. 加载yaml配置文件（所有参数从配置文件读取，无需硬编码）
    tlogger.print(f"Loading config file: {args.c}")
    load_yaml(args, args.c)

    # 3. 检查关键配置（确保测试集路径和保存路径存在）
    if not os.path.exists(args.test_root):
        raise ValueError(f"Test set path not exist! Check 'test_root' in config: {args.test_root}")
    if not os.path.exists(args.save_dir):
        raise ValueError(f"Save directory not exist! Check 'save_dir' in config: {args.save_dir}")

    # 4. 设置设备（GPU优先，无GPU则用CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tlogger.print(f"Using device: {device}")
    

    # 5. 启动测试集推理
    infer_test_set(args, device, tlogger)
    tlogger.print("Inference Script Finished!")
    tlogger.print("=" * 50)