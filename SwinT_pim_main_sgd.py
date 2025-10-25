import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import wandb
import warnings

from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args
from utils.lr_schedule import cosine_decay, adjust_lr, get_lr
from eval import evaluate, cal_train_metrics

warnings.simplefilter("ignore")


def eval_freq_schedule(args, epoch: int):
    """
    根据当前训练的 epoch 调整验证频率（eval_freq）。
    在训练接近尾声时更频繁地进行验证，以便更好地监控模型性能。

    参数:
    args: 包含训练配置参数的对象，其中包括 eval_freq 和 max_epochs。
    epoch: 当前训练的 epoch 数。
    """
    # 如果当前 epoch 大于等于最大训练轮次的 95%，则将验证频率设为 1（每个 epoch 都验证）
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    # 如果当前 epoch 大于等于最大训练轮次的 90% 但小于 95%，同样将验证频率设为 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    # 如果当前 epoch 大于等于最大训练轮次的 80% 但小于 90%，将验证频率设为 2（每两个 epoch 验证一次）
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2


def set_environment(args, tlogger):
    """
    设置训练环境，包括设备、数据加载器、模型、优化器等。

    参数:
    args: 包含训练配置参数的对象。
    tlogger: 用于记录时间日志的对象。

    返回:
    train_loader: 训练数据加载器。
    val_loader: 验证数据加载器。
    model: 构建并初始化的模型。
    optimizer: 优化器（如果仅评估则为None）。
    schedule: 学习率调度器（如果仅评估则为None）。
    scaler: AMP缩放器（如果不使用AMP则为None）。
    amp_context: AMP上下文管理器（如果不使用AMP则是nullcontext）。
    start_epoch: 训练开始的epoch数（如果有预训练模型，则从该模型的epoch开始）。
    """

    print("Setting Environment...")

    # 设置训练设备：如果CUDA可用则使用GPU，否则使用CPU
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### = = = =  Dataset and Data Loader = = = =
    # 构建训练和验证数据加载器
    tlogger.print("Building Dataloader....")

    train_loader, val_loader = build_loader(args)

    # 检查是否成功构建了数据加载器
    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    # 打印训练集信息
    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader)))
    else:
        # raise ValueError("Build train loader fail, please provide legal path.")
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")

    # 打印验证集信息
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()

    ### = = = =  Model = = = =
    # 构建模型
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )  # about return_nodes, we use our default setting

    # 如果提供了预训练模型，则加载权重
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print(start_epoch)
    else:
        start_epoch = 0

    # 将模型移动到指定设备
    model.to(args.device)
    tlogger.print()

    """
    如果你有多GPU设备，可以在单机多GPU情况下使用torch.nn.DataParallel，
    或者使用torch.nn.parallel.DistributedDataParallel实现多进程并行。
    更多详情：https://pytorch.org/tutorials/beginner/dist_overview.html
    """

    # 如果没有训练数据加载器，只进行评估，返回部分对象
    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None, start_epoch

    ### = = = =  Optimizer = = = =
    # 构建优化器
    tlogger.print("Building Optimizer....")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9,
                                    weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    # 如果有预训练模型，加载优化器状态
    if args.pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()

    # 构建学习率调度器
    schedule = cosine_decay(args, len(train_loader))

    # 如果使用混合精度训练(AMP)，设置相关的组件
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    # 返回所有构建的组件
    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch


def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader):
    """
    训练函数，在每个epoch中迭代训练数据并更新模型参数。

    参数:
    args: 包含训练配置参数的对象。
    epoch: 当前训练的 epoch 数。
    model: 要训练的模型。
    scaler: AMP缩放器（如果不使用AMP则为None）。
    amp_context: AMP上下文管理器（如果不使用AMP则是nullcontext）。
    optimizer: 优化器。
    schedule: 学习率调度器。
    train_loader: 训练数据加载器。
    """

    # 清空优化器的梯度
    optimizer.zero_grad()

    # 获取总批次数，仅用于日志记录
    total_batchs = len(train_loader)

    # 定义训练进度显示点（0%, 10%, ..., 100%）
    show_progress = [x / 10 for x in range(11)]
    progress_i = 0

    # 遍历训练数据加载器中的每个批次
    for batch_id, (ids, datas, labels) in enumerate(train_loader):
        # 设置模型为训练模式
        model.train()

        """ = = = = adjust learning rate = = = = """
        # 计算当前迭代次数
        iterations = epoch * len(train_loader) + batch_id
        # 调整学习率
        adjust_lr(iterations, optimizer, schedule)

        # 获取当前批次的样本数量
        batch_size = labels.size(0)

        """ = = = = forward and calculate loss = = = = """
        # 将数据和标签移动到指定设备
        datas, labels = datas.to(args.device), labels.to(args.device)

        # 使用AMP上下文进行前向传播（如果启用AMP）
        with amp_context():
            """
            [Model Return]
                FPN + Selector + Combiner --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1', 'comb_outs'
                FPN + Selector --> return 'layer1', 'layer2', 'layer3', 'layer4', ...(depend on your setting)
                    'preds_0', 'preds_1'
                FPN --> return 'layer1', 'layer2', 'layer3', 'layer4' (depend on your setting)
                ~ --> return 'ori_out'

            [Retuen Tensor]
                'preds_0': logit has not been selected by Selector.
                'preds_1': logit has been selected by Selector.
                'comb_outs': The prediction of combiner.
            """
            # 前向传播获取输出
            outs = model(datas)

            # 初始化总损失
            loss = 0.

            # 遍历模型输出的各个部分，计算相应的损失
            for name in outs:
                # 处理选择器的输出
                if "select_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_s != 0:
                        # 计算选择器损失
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit,
                                                       labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                    else:
                        loss_s = 0.0

                # 处理丢弃部分的输出
                elif "drop_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")

                    if args.lambda_n != 0:
                        # 计算负样本损失
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, args.num_classes]) - 1
                        labels_0 = labels_0.to(args.device)
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                    else:
                        loss_n = 0.0

                # 处理FPN层的输出
                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        # 计算FPN基础损失
                        ### here using 'layer1'~'layer4' is default setting, you can change to your own
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                    else:
                        loss_b = 0.0

                # 处理组合器的输出
                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")

                    if args.lambda_c != 0:
                        # 计算组合器损失
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += args.lambda_c * loss_c

                # 处理原始输出
                elif "ori_out" in name:
                    # 计算原始输出损失
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori

            # 对损失进行平均化处理
            loss /= args.update_freq

        """ = = = = calculate gradient = = = = """
        # 计算梯度（根据是否使用AMP选择不同的方式）
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        """ = = = = update model = = = = """
        # 更新模型参数（每隔update_freq个批次更新一次）
        if (batch_id + 1) % args.update_freq == 0:
            if args.use_amp:
                # 使用AMP更新模型
                scaler.step(optimizer)
                scaler.update()  # next batch
            else:
                # 正常更新模型
                optimizer.step()
            # 清空梯度
            optimizer.zero_grad()

        """ log (MISC) """
        # 记录训练日志（如果启用wandb且达到记录频率）
        if args.use_wandb and ((batch_id + 1) % args.log_freq == 0):
            # 切换到评估模式进行日志记录
            model.eval()
            msg = {}
            msg['info/epoch'] = epoch + 1
            msg['info/lr'] = get_lr(optimizer)
            # 计算并记录训练指标
            cal_train_metrics(args, msg, outs, labels, batch_size)
            # 将日志信息发送到wandb
            wandb.log(msg)

        # 显示训练进度
        train_progress = (batch_id + 1) / total_batchs
        # print(train_progress, show_progress[progress_i])
        if train_progress > show_progress[progress_i]:
            print(".." + str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1

def main(args, tlogger):
    """
    主训练循环函数，负责整个训练和验证过程，包括模型保存（last.pt 和 best.pt）。

    参数:
    args: 包含训练配置参数的对象。
    tlogger: 用于记录时间日志的对象。
    """

    # 调用set_environment函数设置训练环境，获取数据加载器、模型、优化器等
    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(args,
                                                                                                             tlogger)

    # 初始化最佳准确率和最佳评估名称
    best_acc = 0.0
    best_eval_name = "null"

    # 如果启用wandb，则初始化wandb项目并设置初始摘要信息
    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   name=args.exp_name,
                   config=args)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_eval_name"] = best_eval_name
        wandb.run.summary["best_epoch"] = 0

    # 开始训练循环，从start_epoch到max_epochs
    for epoch in range(start_epoch, args.max_epochs):

        """
        训练阶段
        """
        # 如果存在训练数据加载器，则进行训练
        if train_loader is not None:
            tlogger.print("Start Training {} Epoch".format(epoch + 1))
            # 调用train函数进行一个epoch的训练
            train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader)
            tlogger.print()
        else:
            # 如果没有训练数据加载器（仅评估模式），则调用eval_and_save进行评估并保存结果，然后退出循环
            from eval import eval_and_save
            eval_and_save(args, model, val_loader)
            break

        # 根据当前epoch调整验证频率
        eval_freq_schedule(args, epoch)

        # 准备要保存的模型检查点（处理多GPU情况）
        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        # 保存最新的模型检查点
        torch.save(checkpoint, args.save_dir + "backup/last.pt")

        # 根据评估频率进行验证（每个epoch或每隔几个epoch）
        if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
            """
            验证阶段
            """
            acc = -1
            # 如果存在验证数据加载器，则进行验证
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                # 调用evaluate函数进行验证，获取准确率等信息
                acc, eval_name, accs = evaluate(args, model, val_loader)
                # 打印当前验证结果和历史最佳准确率
                tlogger.print("....BEST_ACC: {}% ({}%)".format(max(acc, best_acc), acc))
                tlogger.print()

            # 如果启用wandb，则记录验证指标
            if args.use_wandb:
                wandb.log(accs)

            # 如果当前准确率优于历史最佳准确率，则更新最佳准确率并保存最佳模型
            if acc > best_acc:
                best_acc = acc
                best_eval_name = eval_name
                torch.save(checkpoint, args.save_dir + "backup/best.pt")
            # 如果启用wandb，则更新wandb中的最佳指标摘要
            if args.use_wandb:
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_eval_name"] = best_eval_name
                wandb.run.summary["best_epoch"] = epoch + 1


if __name__ == "__main__":
    # 创建一个时间记录器实例，用于记录和打印时间相关的日志
    tlogger = timeLogger()

    # 打印正在读取配置文件的信息
    tlogger.print("Reading Config...")

    # 获取命令行参数，这些参数包括配置文件路径等
    args = get_args()

    # 断言确保提供了配置文件（.yaml格式），如果没有提供则抛出错误信息
    assert args.c != "", "Please provide config file (.yaml)"

    # 加载指定的YAML配置文件，并将配置内容存入args对象中
    load_yaml(args, args.c)

    # 根据配置创建记录文件夹，用于保存训练过程中的日志、模型等文件
    build_record_folder(args)

    # 打印空行，起到分隔日志的作用
    tlogger.print()

    # 调用main函数开始执行主要的训练或评估流程，传入解析后的参数和时间记录器
    main(args, tlogger)