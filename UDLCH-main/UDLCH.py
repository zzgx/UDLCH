import os
import pdb
import random as rn
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import nets as models
from NCE.DER_Buffer import Buffer
from NCE.DistenceNCE import DistenceNCE
from NCE.NCECriterion import NCESoftmaxLoss
from src.Mat_5 import CMDataset
from utils.config import args
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda'
formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S")
seed = 2025
print(f'seed={seed}')
np.random.seed(seed)
rn.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = True

device_ids = [0, 1]
teacher_device_id = [0, 1]
best_acc = 0
start_epoch = 0
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.pretrain_dir)
os.makedirs(args.ckpt_dir, exist_ok=True)

def main():
    print('===> Preparing data ..')
    print(f'args.category_split_ratio={args.category_split_ratio}')
    print(f'args.train_batch_size={args.train_batch_size}')
    print(f'args.bit={args.bit}')
    print(f'args.buffer_size={args.buffer_size}')
    print(f'args.K={args.K}')
    print(f'args.alpha={args.alpha}')
    print(f'args.der_a={args.der_a}')
    print(f'args.der_b={args.der_b}')
    dataset = CMDataset(args.data_name, batch_size=args.train_batch_size, category_split_ratio=args.category_split_ratio)


    task_sequence = [dataset.visible_set, dataset.invisible_set]

    task_loaders = [DataLoader(task, batch_size=args.train_batch_size, shuffle=False) for task in task_sequence]


    visible_retrieval_loader = DataLoader(dataset.visible_retrieval_set, batch_size=args.train_batch_size, shuffle=False)
    visible_query_loader = DataLoader(dataset.visible_query_set, batch_size=args.train_batch_size, shuffle=False)

    invisible_retrieval_loader = DataLoader(dataset.invisible_retrieval_set, batch_size=args.train_batch_size, shuffle=False)
    invisible_query_loader = DataLoader(dataset.invisible_query_set, batch_size=args.train_batch_size, shuffle=False)
    print('===> Building ImageNet and TextNet..')

    if 'fea' in args.data_name:

        image_model = models.__dict__['ImageNet'](y_dim=4096, bit=args.bit, hiden_layer=args.num_hiden_layers[0]).cuda() # 4096 y_dim=train_dataset.imgs.shape[1]
        backbone = None
    else:
        backbone = models.__dict__[args.arch](pretrained=args.pretrain, feature=True).cuda()  # 深度特征提取的骨干网络（backbone），（动态加载模型（如 resnet 或 vgg））pretrained=args.pretrain
        if 'vgg' in args.arch.lower():
            fea_dim = 4096
        elif args.arch == 'resnet18' or args.arch == 'resnet34':
            fea_dim = 512
        else:
            fea_dim = 2048
        fea_net = models.__dict__['ImageNet'](y_dim=fea_dim, bit=args.bit, hiden_layer=args.num_hiden_layers[0]).cuda()
        image_model = nn.Sequential(backbone, fea_net)
    if 'mirflickr' in args.data_name:
        text_dim = 1386
    elif 'nuswide' in args.data_name:
        text_dim = 1000
    elif 'iapr' in args.data_name:
        text_dim = 2912
    elif 'mscoco' in args.data_name:
        text_dim = 300
    text_model = models.__dict__['TextNet'](y_dim=text_dim , bit=args.bit, hiden_layer=args.num_hiden_layers[1]).cuda()  # y_dim=train_dataset.text_dim| flickr25k的y_dim=1386 |ODIR=102
    parameters = list(image_model.parameters()) + list(text_model.parameters())  # 模型参数



    wd = args.wd
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=wd)


    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        image_model.load_state_dict(ckpt['image_model_state_dict'])
        text_model.load_state_dict(ckpt['text_model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')


    def set_train():
        image_model.train()
        text_model.train()
        if backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            for name, param in backbone.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

    def set_eval():
        image_model.eval()
        text_model.eval()

    global patience_counter
    def early_stopping(current_accuracy, best_accuracy, patience=10):

        global patience_counter
        if current_accuracy > best_accuracy:
            patience_counter = 0
        else:
            patience_counter += 1
        print(f'patience_counter={patience_counter}')
        if patience_counter >= patience:
            print(f"Training stopped after {patience} epochs without improvement.")
            patience_counter = 0
            return True
        else:
            return False

    n_data = len(dataset)
    print(f'n_data={n_data}')
    contrast = DistenceNCE(args.bit, n_data, args.K, args.T, args.momentum).cuda()
    criterion_contrast = NCESoftmaxLoss().cuda()
    buffer = Buffer(args.buffer_size, 'cuda')


    def train():
        set_train()
        base_max_avg = 0.
        base_max_t2i = 0.
        base_max_i2t = 0.
        increment_max_avg = 0.
        increment_max_t2i = 0.
        increment_max_i2t = 0.
        for task, tr_loader in enumerate(task_loaders):
            count = len(tr_loader)

            if task == 1 and (args.category_split_ratio == (24, 0) or args.category_split_ratio == (10, 0) or args.category_split_ratio == (255, 0) or args.category_split_ratio == (80, 0)):
                print("Base only! End.")
                pass
            else:
                if task == 1:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr * 0.1
                for n in range(int(args.task_epochs)):
                    if n == args.warmup_epoch:
                        print('using hash memory bank now...')
                    train_loss = 0.
                    for batch_idx, (idx, images, texts, tagets) in enumerate(tr_loader):
                        back_loss = 0.
                        optimizer.zero_grad()
                        images, texts, idx = [img.cuda() for img in images], [txt.cuda() for txt in texts], [idx.cuda()]

                        images_outputs = [image_model(im) for im in images]
                        texts_outputs = [text_model(txt.float()) for txt in texts]

                        out_l, out_ab = contrast(torch.cat(images_outputs), torch.cat(texts_outputs), torch.cat(idx),epoch=n - args.warmup_epoch)
                        l_loss = criterion_contrast(out_l)
                        ab_loss = criterion_contrast(out_ab)
                        Lc = l_loss + ab_loss
                        Lc = Lc * args.alpha
                        train_loss += Lc.item()

                        # DER loss
                        if not buffer.is_empty():
                            buf_idx, buf_imgs, buf_tags, buf_IMGlogits, buf_TAGlogits = buffer.get_data(args.train_batch_size)

                            buf_img_outputs = [image_model(buf_imgs)]
                            buf_tag_outputs = [text_model(buf_tags.float())]

                            buf_img_outputs = torch.cat(buf_img_outputs)
                            buf_tag_outputs = torch.cat(buf_tag_outputs)

                            der_img_loss = F.mse_loss(buf_img_outputs, buf_IMGlogits)
                            der_tag_loss = F.mse_loss(buf_tag_outputs, buf_TAGlogits)

                            DER_loss = args.der_b * der_img_loss + (1-args.der_b) * der_tag_loss
                            DER_loss = args.der_a * DER_loss
                            train_loss += DER_loss.item()

                        if batch_idx == 0:
                            back_loss = Lc
                        else:
                            back_loss = Lc + DER_loss

                        back_loss.backward()
                        optimizer.step()

                        buffer.add_data(idx=torch.cat(idx), imgs=torch.cat(images), tags=torch.cat(texts), img_logits=torch.cat(images_outputs).data, tag_logits=torch.cat(texts_outputs).data)

                        clip_grad_norm_(parameters, 1.)

                    print(f'epoch={n}')

                    if task == 0:
                        print('Base hash representation Avg Loss: %.3f | LR: %g' % (train_loss / count, optimizer.param_groups[0]['lr']))
                        i2t, t2i, avg = eval(visible_retrieval_loader, visible_query_loader)
                        stop_training = early_stopping(avg, base_max_avg)
                        if stop_training:
                            break
                        print(f'Base mAP: i2t={i2t:.3f}   t2i={t2i:.3f}   avg={avg:.3f}')

                        if avg > base_max_avg:
                            base_max_avg = avg
                            base_max_i2t = i2t
                            base_max_t2i = t2i
                            print(f'Saving Base Max mAP : i2t={base_max_i2t:.3f}   t2i={base_max_t2i:.3f}   avg={base_max_avg:.3f}')
                            state = {
                                'image_model_state_dict': image_model.state_dict(),
                                'text_model_state_dict': text_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'Avg': base_max_avg,
                                'Img2Txt': base_max_i2t,
                                'Txt2Img': base_max_t2i,
                            }
                            torch.save(state,os.path.join(args.ckpt_dir, 'UDLCH_%d_%d_%d_base_best_checkpoint.t7' % (args.category_split_ratio[0], args.category_split_ratio[1], args.bit)))
                    else:
                        print('Increment hash representation Avg Loss: %.3f | LR: %g' % (train_loss / count, optimizer.param_groups[0]['lr']))
                        i2t, t2i, avg = eval(invisible_retrieval_loader, invisible_query_loader)
                        print(f'Increment mAP: i2t={i2t:.3f}   t2i={t2i:.3f}   avg={avg:.3f}')
                        if avg > increment_max_avg:
                            increment_max_avg = avg
                            increment_max_i2t = i2t
                            increment_max_t2i = t2i
                            print(f'Saving Increment Max mAP : i2t={increment_max_i2t:.3f}   t2i={increment_max_t2i:.3f}   avg={increment_max_avg:.3f}')
                            state = {
                                'image_model_state_dict': image_model.state_dict(),
                                'text_model_state_dict': text_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'Avg': increment_max_avg,
                                'Img2Txt': increment_max_i2t,
                                'Txt2Img': increment_max_t2i,
                            }
                            torch.save(state,os.path.join(args.ckpt_dir, 'UDLCH_%d_%d_%d_increment_best_checkpoint.t7' % (args.category_split_ratio[0], args.category_split_ratio[1], args.bit)))

        print()
        print(f'Base Max mAP : i2t={base_max_i2t:.3f}   t2i={base_max_t2i:.3f}  avg={base_max_avg:.3f}')
        print(f'Increment Max mAP : i2t={increment_max_i2t:.3f}   t2i={increment_max_t2i:.3f}   avg={increment_max_avg:.3f}')
        if args.category_split_ratio == (24, 0) or args.category_split_ratio == (10, 0) or args.category_split_ratio == (255, 0) or args.category_split_ratio == (80, 0):
            ckpt = torch.load(os.path.join(args.ckpt_dir, 'UDLCH_%d_%d_%d_base_best_checkpoint.t7' % (args.category_split_ratio[0], args.category_split_ratio[1], args.bit)))
            image_model.load_state_dict(ckpt['image_model_state_dict'])
            text_model.load_state_dict(ckpt['text_model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            eval(visible_retrieval_loader, visible_query_loader,PR=True)
        else:
            ckpt = torch.load(os.path.join(args.ckpt_dir, 'UDLCH_%d_%d_%d_increment_best_checkpoint.t7' % (args.category_split_ratio[0], args.category_split_ratio[1], args.bit)))
            image_model.load_state_dict(ckpt['image_model_state_dict'])
            text_model.load_state_dict(ckpt['text_model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            increment_base_i2t, increment_base_t2i, increment_base_avg = eval(visible_retrieval_loader, visible_query_loader)
            print(f'Increment_Base Max mAP : i2t={increment_base_i2t:.3f}   t2i={increment_base_t2i:.3f}   avg={increment_base_avg:.3f}')
            print(f'forgetting={(base_max_avg-increment_base_avg)*100:.4f}%')

    def eval(retrieval_loader, query_loader, PR=False):
        set_eval()
        imgs, txts, labs = [], [], []
        imgs_te, txts_te, labs_te = [], [], []
        with torch.no_grad():
            for batch_idx, (idx, images, texts, targets) in enumerate(retrieval_loader):
                images, texts, targets = [img.cuda() for img in images], [txt.cuda() for txt in texts], targets.cuda()
                images_outputs = [image_model(im) for im in images]
                texts_outputs = [text_model(txt.float()) for txt in texts]
                imgs += images_outputs
                txts += texts_outputs
                labs.append(targets)
            retrieval_imgs = torch.cat(imgs).sign_().cuda()
            retrieval_txts = torch.cat(txts).sign_().cuda()
            retrieval_labs = torch.cat(labs).cuda()

            for batch_idx, (idx, images, texts, targets) in enumerate(query_loader):
                images, texts, targets = [img.cuda() for img in images], [txt.cuda() for txt in texts], targets.cuda()
                images_outputs = [image_model(im) for im in images]
                texts_outputs = [text_model(txt.float()) for txt in texts]
                imgs_te += images_outputs
                txts_te += texts_outputs
                labs_te.append(targets)
            query_imgs = torch.cat(imgs_te).sign_().cuda()
            query_txts = torch.cat(txts_te).sign_().cuda()
            query_labs = torch.cat(labs_te).cuda()

            i2t = calculate_top_map(query_imgs, retrieval_txts, query_labs, retrieval_labs, topk=0)
            t2i = calculate_top_map(query_txts, retrieval_imgs, query_labs, retrieval_labs, topk=0)

            avg = (i2t + t2i) / 2.
            # if PR:
            #     i2t_P, i2t_R = calculate_pr_curve(query_imgs, retrieval_txts, query_labs, retrieval_labs)
            #     print(f'i2t_P={i2t_P}')
            #     print(f'i2t_R={i2t_R}')
            #     plot_pr_curve(i2t_P,i2t_R)
            #     t2i_P, t2i_R = calculate_pr_curve(query_txts, retrieval_imgs, query_labs, retrieval_labs)
            #     print(f't2i_P={t2i_P}')
            #     print(f't2i_R={t2i_R}')
            #     plot_pr_curve(t2i_P, t2i_R)
            #
            #     timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            #     PR_save_dir = "logs/UDLCH"
            #     PR_file_name = f"UDLCH_pr_{args.data_name}_{args.bit}bit_{timestamp}.json"
            #     PR_file_path = os.path.join(PR_save_dir, PR_file_name)
            #     os.makedirs(PR_save_dir, exist_ok=True)
            #     with open(PR_file_path, "w") as f:
            #         json.dump({"i2t_P": i2t_P.tolist(), "i2t_R": i2t_R.tolist(), "t2i_P": t2i_P.tolist(), "t2i_R": t2i_R.tolist()}, f)

        return i2t, t2i, avg

    for epoch in range(start_epoch, args.max_epochs):
        train()


def calculate_hamming(B1, B2):
    leng = B2.size(1)
    dot = torch.matmul(B1.unsqueeze(0), B2.t()).squeeze(0)
    distH = 0.5 * (leng - dot)
    return distH

def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):

    qu_L = qu_L.float()
    re_L = re_L.float()

    if topk == 0:
        topk = re_L.size(0)
    num_query = qu_L.size(0)
    topkmap = torch.tensor(0.0, device=qu_B.device)

    for i in range(num_query):
        gnd = (torch.matmul(qu_L[i], re_L.t()) > 0).float()
        hamm = calculate_hamming(qu_B[i], re_B)
        _, ind = torch.sort(hamm, descending=False)
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = tgnd.sum()
        if tsum.item() == 0:
            continue

        steps = int(tsum.item())
        count = torch.linspace(1, float(steps), steps, device=tgnd.device)

        tindex = torch.nonzero(tgnd, as_tuple=True)[0].float() + 1.0

        topkmap_ = torch.mean(count / tindex)
        topkmap += topkmap_

    topkmap = topkmap / num_query
    return topkmap

def calculate_pr_curve(qB, rB, query_label, retrieval_label):
    query_label = query_label.float()
    retrieval_label = retrieval_label.float()
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R
def plot_pr_curve(P, R, title="Precision-Recall Curve"):
    """
    Plot the Precision-Recall curve.
    :param P: Precision values
    :param R: Recall values
    :param title: Title of the plot
    """
    plt.figure(figsize=(10, 10))
    plt.plot(R, P, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    plt.title(title, fontsize=16)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()