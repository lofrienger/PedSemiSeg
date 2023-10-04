import argparse
import copy
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from dataloaders.dataset import (MultiAugment_polyp, PolypDataset,
                                 TwoStreamBatchSampler, trsf_valid_image_224)
from networks.net_factory import net_factory
from torch.distributions import Categorical
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils import losses, ramps
from val_2D import test_polyp_batch

parser = argparse.ArgumentParser()
parser.add_argument('--ds_root', type=str, default='/mnt/data-hdd/wa/dataset/Polyp/SUN_SEG/data/SUN-SEG')
parser.add_argument('--csv_root', type=str, default='/mnt/data-ssd/wa/SSL4MIS/data/polyp')
parser.add_argument("--method", type=str, default="PedSemiSeg", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=20000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=16, help="batch_size per gpu")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--num_classes", type=int, default=2, help="output channel of network")
parser.add_argument("--conf_thresh", type=float, default=0.8, help="confidence threshold for using pseudo-labels")

parser.add_argument('--comp_loss', type=str, default='True', help='use complement loss or not')
parser.add_argument('--peer_loss', type=str, default='True', help='use peer loss or not')

parser.add_argument("--labeled_bs", type=int, default=8, help="labeled_batch_size per gpu")
parser.add_argument('--labeled_ratio', type=float, default=0.5, help='ratio of labeled data')

parser.add_argument("--consistency_strategy", type=str, default="exp", help="consistency update strategy")
parser.add_argument("--consistency", type=float, default=1, help="consistency")
parser.add_argument('--consistency_rampup', type=float, default=5000.0, help='consistency_rampup')

args = parser.parse_args()

def get_current_consistency_weight(iter):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(iter, args.consistency_rampup)


def train(args, snapshot_path):
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=3, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def get_comp_loss(weak, strong):
        """get complementary loss and adaptive sample weight.
        Compares least likely prediction (from strong augment) with argmin of weak augment.

        Args:
            weak (batch): weakly augmented batch
            strong (batch): strongly augmented batch

        Returns:
            comp_loss, as_weight
        """
        il_output = torch.reshape(
            strong,
            (
                args.batch_size,
                args.num_classes,
                224 * 224,
            ),
        )
        # calculate entropy for image-level preds (tensor of length labeled_bs)
        as_weight = 1 - (Categorical(probs=il_output).entropy() / np.log(224 * 224))

        # batch level average of entropy
        as_weight = torch.mean(as_weight)
        # complementary loss
        comp_labels = torch.argmin(weak.detach(), dim=1, keepdim=False)
        comp_loss = as_weight * ce_loss(
            torch.add(torch.negative(strong), 1),
            comp_labels,
        )
        return comp_loss

    def normalize(tensor):
        min_val = tensor.min(1, keepdim=True)[0]
        max_val = tensor.max(1, keepdim=True)[0]
        result = tensor - min_val
        result = result / max_val
        return result

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    db_train = PolypDataset(ds_root=args.ds_root, csv_root=args.csv_root, split="train", 
                            transform=transforms.Compose([MultiAugment_polyp((224,224))]))
    db_val = PolypDataset(ds_root=args.ds_root, csv_root=args.csv_root, split="valid", transform=trsf_valid_image_224)
    db_test = PolypDataset(ds_root=args.ds_root, csv_root=args.csv_root, split="test", transform=trsf_valid_image_224)

    total_image = len(db_train)
    labeled_image = int(total_image * args.labeled_ratio)
    logging.info("Total images is: {}, labeled images is: {}".format(total_image, labeled_image))
    labeled_idxs = list(range(0, labeled_image))
    unlabeled_idxs = list(range(labeled_image, total_image))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = create_model()

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # set to train
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    iter_num = 0
    start_epoch = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch_train, sampled_batch in enumerate(trainloader):
            weak_batch, strong_batch, strong_batch2, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["image_strong2"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, strong_batch2, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                strong_batch2.cuda(),
                label_batch.cuda(),
            )

            # outputs for model
            outputs_weak = model(weak_batch)
            outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
            outputs_strong = model(strong_batch)
            outputs_strong_soft = torch.softmax(outputs_strong, dim=1)
            outputs_strong2 = model(strong_batch2)
            outputs_strong_soft2 = torch.softmax(outputs_strong2, dim=1)

            # minmax normalization for softmax outputs before applying mask
            pseudo_mask = (normalize(outputs_weak_soft) > args.conf_thresh).float()
            outputs_weak_masked = outputs_weak_soft * pseudo_mask
            pseudo_outputs = torch.argmax(outputs_weak_masked[args.labeled_bs:].detach(), dim=1, keepdim=False)

            # supervised loss
            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )

            pos_loss = (ce_loss(outputs_strong[args.labeled_bs:], pseudo_outputs) +
                        dice_loss(outputs_strong_soft[args.labeled_bs:], pseudo_outputs.unsqueeze(1)) +
                        ce_loss(outputs_strong2[args.labeled_bs:], pseudo_outputs) +
                        dice_loss(outputs_strong_soft2[args.labeled_bs:], pseudo_outputs.unsqueeze(1)))

            # complementary loss and adaptive sample weight for negative learning
            if args.comp_loss == 'True':                    
                # complementary loss and adaptive sample weight for negative learning
                comp_loss1 = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft)
                comp_loss2 = get_comp_loss(weak=outputs_weak_soft, strong=outputs_strong_soft2)
                comp_loss =  comp_loss1 +  comp_loss2
            else:
                comp_loss = 0
            
            # entropy-guided peer loss
            if args.peer_loss == 'True':  
                ent_loss1 = losses.entropy_loss(outputs_strong_soft, C=2)
                ent_loss2 = losses.entropy_loss(outputs_strong_soft2, C=2)

                if ent_loss1 > ent_loss2:
                    pseudo_mask_strong = (normalize(outputs_strong_soft2) > args.conf_thresh).float()
                    outputs_strong_masked = outputs_strong_soft2 * pseudo_mask_strong
                    pseudo_outputs_outputs = torch.argmax(outputs_strong_masked[args.labeled_bs:].detach(), dim=1, keepdim=False)

                    peer_loss = ce_loss(outputs_strong[args.labeled_bs :], pseudo_outputs_outputs) 
                    + dice_loss(outputs_strong_soft[args.labeled_bs :], pseudo_outputs_outputs.unsqueeze(1))
                else:
                    pseudo_mask_strong = (normalize(outputs_strong_soft) > args.conf_thresh).float()
                    outputs_strong_masked = outputs_strong_soft * pseudo_mask_strong
                    pseudo_outputs_outputs = torch.argmax(outputs_strong_masked[args.labeled_bs:].detach(), dim=1, keepdim=False)

                    peer_loss = ce_loss(outputs_strong2[args.labeled_bs :], pseudo_outputs_outputs) 
                    + dice_loss(outputs_strong_soft2[args.labeled_bs :], pseudo_outputs_outputs.unsqueeze(1))
            else:
                peer_loss = 0

                # unsupervised loss
            unsup_loss = (
                pos_loss
                + comp_loss
                + peer_loss
            )

            if args.consistency_strategy != 'const':
                consistency_weight = get_current_consistency_weight(iter_num)
            else:
                consistency_weight = args.consistency

            loss = sup_loss + consistency_weight * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1

            logging.info("iteration %d: total_loss: %f sup_loss: %f unsup_loss: %f" % (iter_num, loss.item(), sup_loss.item(), unsup_loss.item()))

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = np.zeros(4)
                for i_batch_val, sampled_batch in enumerate(valloader):
                    sampled_batch["image"] = sampled_batch["image"].float().cuda()
                    metric_i = test_polyp_batch(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model)
                    metric_list += metric_i
                metric_list = metric_list / (i_batch_val+1)
                metric_dict = dict(dc=metric_list[0], jc=metric_list[1], pre=metric_list[2], hd=metric_list[3])
                logging.info('==> Valid iteration %d: unet metrics: %s.' % (iter_num, metric_dict))

                if metric_dict['dc'] > best_performance:
                    best_performance = metric_dict['dc']
                    best_model = copy.deepcopy(model)
                    best_iter = iter_num
                    # save_mode_path = os.path.join(snapshot_path,
                    #                                 'iter_{}_dice_{}.pth'.format(
                    #                                     iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                                '{}_best_model.pth'.format(args.model))
                    # torch.save(best_model.state_dict(), save_mode_path)
                    torch.save(best_model.state_dict(), save_best)
                    logging.info('==> New best valid dice: %f, at iteration %d' % (metric_dict['dc'], best_iter))
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(
                    snapshot_path, '{}_last_model.pth'.format(args.model))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save last model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    logging.info("Training Finished!")
    logging.info("Best Validation dice: {} in iter: {}.".format(round(best_performance, 4), best_iter))

    logging.info('==> Test with best model:')
    test_metric = np.zeros(4)
    for i_batch_test, sampled_batch in enumerate(testloader):
        sampled_batch['image'] = sampled_batch['image'].float().cuda()
        metric_i = test_polyp_batch(
            sampled_batch['image'], sampled_batch['label'], best_model)
        test_metric += metric_i
    test_metric = test_metric / (i_batch_test+1)
    metric_dict = dict(dc=test_metric[0], jc=test_metric[1], pre=test_metric[2], hd=test_metric[3])
    logging.info('==> Test metrics: %s' % metric_dict)


if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    snapshot_path = "../model/polyp_pedsemiseg/{}/seed_{}/{}/r{}".format(
        args.model, args.seed, args.method, args.labeled_ratio)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    train(args, snapshot_path)
    