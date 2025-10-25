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
from eval import evaluate, cal_train_metrics, _average_top_k_result  # 确保导入_average_top_k_result

warnings.simplefilter("ignore")


def eval_freq_schedule(args, epoch: int):
    if epoch >= args.max_epochs * 0.95:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.9:
        args.eval_freq = 1
    elif epoch >= args.max_epochs * 0.8:
        args.eval_freq = 2


def set_environment(args, tlogger):
    print("Setting Environment...")
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Dataset and Data Loader
    tlogger.print("Building Dataloader....")
    train_loader, val_loader = build_loader(args)

    if train_loader is None and val_loader is None:
        raise ValueError("Find nothing to train or evaluate.")

    if train_loader is not None:
        print("    Train Samples: {} (batch: {})".format(len(train_loader.dataset), len(train_loader)))
    else:
        print("    Train Samples: 0 ~~~~~> [Only Evaluation]")
    if val_loader is not None:
        print("    Validation Samples: {} (batch: {})".format(len(val_loader.dataset), len(val_loader)))
    else:
        print("    Validation Samples: 0 ~~~~~> [Only Training]")
    tlogger.print()

    ### Model
    tlogger.print("Building Model....")
    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )

    start_epoch = 0
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], strict=False)
        start_epoch = checkpoint['epoch']
        print(f"Loaded pretrained model from epoch {start_epoch}")

    model.to(args.device)
    tlogger.print()

    if train_loader is None:
        return train_loader, val_loader, model, None, None, None, None, start_epoch

    ### Optimizer
    tlogger.print("Building Optimizer....")
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr, nesterov=True, momentum=0.9,
                                    weight_decay=args.wdecay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

    if args.pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    tlogger.print()
    schedule = cosine_decay(args, len(train_loader))

    ### AMP
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
        amp_context = torch.cuda.amp.autocast
    else:
        scaler = None
        amp_context = contextlib.nullcontext

    return train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch


def train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader):
    """修正训练Acc计算逻辑，打印训练Loss和Acc"""
    optimizer.zero_grad()
    total_batchs = len(train_loader)
    show_progress = [x / 10 for x in range(11)]
    progress_i = 0

    # 记录整个epoch的训练指标
    epoch_total_loss = 0.0  # 总训练Loss
    epoch_correct = 0  # 总正确样本数（用于计算平均Acc）
    total_samples = 0  # 总训练样本数

    for batch_id, (ids, datas, labels) in enumerate(train_loader):
        model.train()
        batch_size = labels.size(0)
        total_samples += batch_size
        labels = labels.to(args.device)  # 移动标签到设备

        ### 调整学习率
        iterations = epoch * len(train_loader) + batch_id
        adjust_lr(iterations, optimizer, schedule)

        ### 前向传播与损失计算
        datas = datas.to(args.device)
        batch_loss = 0.0  # 当前batch的总Loss

        with amp_context():
            outs = model(datas)
            loss = 0.

            # 计算各部分Loss（与原有逻辑一致）
            for name in outs:
                if "select_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_s != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        loss_s = nn.CrossEntropyLoss()(logit, labels.unsqueeze(1).repeat(1, S).flatten(0))
                        loss += args.lambda_s * loss_s
                        batch_loss += loss_s.item() * args.lambda_s

                elif "drop_" in name:
                    if not args.use_selection:
                        raise ValueError("Selector not use here.")
                    if args.lambda_n != 0:
                        S = outs[name].size(1)
                        logit = outs[name].view(-1, args.num_classes).contiguous()
                        n_preds = nn.Tanh()(logit)
                        labels_0 = torch.zeros([batch_size * S, args.num_classes]).to(args.device) - 1
                        loss_n = nn.MSELoss()(n_preds, labels_0)
                        loss += args.lambda_n * loss_n
                        batch_loss += loss_n.item() * args.lambda_n

                elif "layer" in name:
                    if not args.use_fpn:
                        raise ValueError("FPN not use here.")
                    if args.lambda_b != 0:
                        loss_b = nn.CrossEntropyLoss()(outs[name].mean(1), labels)
                        loss += args.lambda_b * loss_b
                        batch_loss += loss_b.item() * args.lambda_b

                elif "comb_outs" in name:
                    if not args.use_combiner:
                        raise ValueError("Combiner not use here.")
                    if args.lambda_c != 0:
                        loss_c = nn.CrossEntropyLoss()(outs[name], labels)
                        loss += args.lambda_c * loss_c
                        batch_loss += loss_c.item() * args.lambda_c

                elif "ori_out" in name:
                    loss_ori = F.cross_entropy(outs[name], labels)
                    loss += loss_ori
                    batch_loss += loss_ori.item()

            # 梯度累积：还原真实Loss
            loss /= args.update_freq
            batch_real_loss = loss.item() * args.update_freq  # 当前batch的真实Loss
            epoch_total_loss += batch_real_loss

        ### 反向传播与参数更新
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_id + 1) % args.update_freq == 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        ### 计算并打印当前batch的训练Acc和Loss
        if (batch_id + 1) % args.log_freq == 0:
            # 计算当前batch的Acc（优先取combiner，无则取原始输出）
            if args.use_combiner and "comb_outs" in outs:
                pred = torch.argmax(outs["comb_outs"], dim=1)
            elif "ori_out" in outs:
                pred = torch.argmax(outs["ori_out"], dim=1)
            else:
                # 若以上都没有，取最后一个FPN层的输出
                pred = torch.argmax(outs["layer4"].mean(1), dim=1)
            
            # 计算当前batch的正确数和Acc
            batch_correct = (pred == labels).sum().item()
            batch_acc = (batch_correct / batch_size) * 100  # 转换为百分比
            epoch_correct += batch_correct  # 累加至总正确数

            # 打印batch级指标
            print(f"[Train] Epoch {epoch+1:2d} | Batch {batch_id+1:4d}/{total_batchs:4d} | "
                  f"Loss: {batch_real_loss:.4f} | Acc: {batch_acc:.2f}%")

        ### 显示训练进度
        train_progress = (batch_id + 1) / total_batchs
        if train_progress > show_progress[progress_i]:
            print(".." + str(int(show_progress[progress_i] * 100)) + "%", end='', flush=True)
            progress_i += 1

    ### 打印当前epoch的训练汇总
    avg_train_loss = epoch_total_loss / total_batchs  # 平均Loss（按batch数）
    avg_train_acc = (epoch_correct / total_samples) * 100  # 平均Acc（按样本数）
    print(f"\n[Train Summary] Epoch {epoch+1:2d} | Avg Loss: {avg_train_loss:.4f} | Avg Acc: {avg_train_acc:.2f}%")


def main(args, tlogger):
    train_loader, val_loader, model, optimizer, schedule, scaler, amp_context, start_epoch = set_environment(args, tlogger)

    best_acc = 0.0
    best_eval_name = "null"

    if args.use_wandb:
        wandb.init(entity=args.wandb_entity, project=args.project_name, name=args.exp_name, config=args)
        wandb.run.summary["best_acc"] = best_acc
        wandb.run.summary["best_epoch"] = 0

    for epoch in range(start_epoch, args.max_epochs):
        ### 训练阶段
        if train_loader is not None:
            tlogger.print("Start Training {} Epoch".format(epoch + 1))
            train(args, epoch, model, scaler, amp_context, optimizer, schedule, train_loader)
            tlogger.print()
        else:
            from eval import eval_and_save
            eval_and_save(args, model, val_loader)
            break

        ### 调整验证频率
        eval_freq_schedule(args, epoch)

        ### 保存最新模型
        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(checkpoint, args.save_dir + "backup/last.pt")

        ### 验证阶段（恢复原始逻辑，不计算验证Loss）
        if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
            acc = -1
            if val_loader is not None:
                tlogger.print("Start Evaluating {} Epoch".format(epoch + 1))
                # 恢复原始evaluate调用（仅返回3个值）
                acc, eval_name, accs = evaluate(args, model, val_loader)
                # 打印验证Acc（不含Loss）
                print(f"[Val] Epoch {epoch+1:2d} | Best Acc: {max(acc, best_acc):.2f}% (Current Acc: {acc:.2f}%)")
                tlogger.print()

            ### 更新wandb日志
            if args.use_wandb:
                wandb.log(accs)

            ### 更新最佳模型
            if acc > best_acc:
                best_acc = acc
                best_eval_name = eval_name
                torch.save(checkpoint, args.save_dir + "backup/best.pt")
                print(f"[Update Best Model] Epoch {epoch+1:2d} | Best Acc: {best_acc:.2f}%")

            ### 更新wandb摘要
            if args.use_wandb:
                wandb.run.summary["best_acc"] = best_acc
                wandb.run.summary["best_epoch"] = epoch + 1


if __name__ == "__main__":
    tlogger = timeLogger()
    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    if not hasattr(args, "log_freq"):
        args.log_freq = 10  # 默认每10个batch打印一次
    build_record_folder(args)
    tlogger.print()
    main(args, tlogger)
