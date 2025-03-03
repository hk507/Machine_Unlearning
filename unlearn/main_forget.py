import copy
import os
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import sys
import arg_parser
import evaluation
import pruner
import unlearn
import utils
from trainer import validate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class Logger(object): # log
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
def main():
    # 解析命令行参数
    args = arg_parser.parse_args()
    sys.stdout = Logger(
        f"logs/{args.unlearn}--{args.dataset} --{args.num_indexes_to_replace} --{args.unlearn_epochs} --{args.unlearn_lr}.txt")
    # 设置设备（CPU/GPU）
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 设置随机种子，确保结果的可重复性
    if args.seed:
        # if args.unlearn == "retrain":
        #     args.seed = random.randint(0, 1000)
        # else:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    # 准备数据集和模型
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    # 创建数据加载器
    def replace_loader_dataset(
            dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    # 根据数据集的不同（如SVHN），对标记的数据集进行处理，分离出需要“遗忘”的数据和保留的数据
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print("遗忘数据集大小", len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print("剩余数据集大小", len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
    # 创建数据加载器，用于训练、验证和测试
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
 
    evaluation_result = None
    # 加载模型的权重
    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)
    # 根据参数选择“遗忘”方法，并执行“遗忘”过程
    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            print("state_dict")
            checkpoint = checkpoint["state_dict"]
        masks = torch.load(args.masks, map_location=device)
        if "mask" in masks.keys():
            print("mask")
            masks = masks["mask"]
        # if (
        #         args.unlearn == "retrain"
        # ):
        #
        #     pruner.prune_model_custom(model, masks)
        pruner.sample_model_custom(model, masks)

        # current_mask = pruner.extract_mask(checkpoint)
        # pruner.prune_model_custom(model, current_mask)
        pruner.check_sparsity(model)

        if (
                args.unlearn != "retrain"
                and args.unlearn != "retrain_sam"
                and args.unlearn != "retrain_ls"
        ):
            model.load_state_dict(checkpoint, strict=False)
            params = count_parameters(model)
            print(f'The model has {params} parameters.')
            model.eval()
            print("验证模型在遗忘前的准确性:")
            initial_accuracy = {}
            for name, loader in unlearn_data_loaders.items():
                utils.dataset_convert_to_test(loader.dataset, args)
                val_acc = validate(loader, model, criterion, args)
                initial_accuracy[name] = val_acc
                print(f"{name} acc before forgetting: {val_acc}")
        unlearn_method = unlearn.get_unlearn_method(args.unlearn)

        unlearn_method(unlearn_data_loaders, model, criterion, args)

        unlearn.save_unlearn_checkpoint(model, None, args)
        pruner.check_sparsity(model)

    if evaluation_result is None:
        evaluation_result = {}
    # 如果没有预先存在的评估结果，执行验证过程并计算准确率
    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")
        # 保存评估结果和模型的“遗忘”状态
        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    # 执行额外的评估，如遗忘效果和训练隐私评估
    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    """training privacy MIA:
        in distribution: retain
        out of distribution: test
        target: (retain, test)"""
    if "SVC_MIA_training_privacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        retain_len = len(retain_dataset)
        num = test_len // 2

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
        target_train = torch.utils.data.Subset(
            retain_dataset, list(range(num, retain_len))
        )
        shadow_test = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
        target_test = torch.utils.data.Subset(
            test_loader.dataset, list(range(num, test_len))
        )

        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=args.batch_size, shuffle=False
        )

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=args.batch_size, shuffle=False
        )
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=shadow_test_loader,
            target_train=target_train_loader,
            target_test=target_test_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
