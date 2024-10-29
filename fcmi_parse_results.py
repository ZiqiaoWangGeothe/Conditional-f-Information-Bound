from tqdm import tqdm
import os
import argparse
import pickle

import numpy as np
import torch

from nnlib.nnlib import utils
from nnlib.nnlib.matplotlib_utils import import_matplotlib

matplotlib, plt = import_matplotlib()

from modules.bound_utils import estimate_fcmi_bound_classification, estimate_sgld_bound, \
    estimate_kl_bound_classification, estimate_lg_bound_classification, estimate_interp_bound_classification, estimate_ecmi_bound_classification
from scripts.fcmi_train_classifier import mnist_ld_schedule, \
    cifar_resnet50_ld_schedule  # for pickle to be able to load LD methods
import methods


class NestedDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


def compute_acc(preds, mask, dataset):
    labels = [y for x, y in dataset]
    labels = torch.tensor(labels).long()
    indices = 2*np.arange(len(mask)) + mask
    acc = (preds[indices].argmax(dim=1) == labels[indices]).float().mean()
    return utils.to_numpy(acc)
    
def compute_error(preds, num, dataset):
    labels = [y for x, y in dataset]
    labels = torch.tensor(labels).long()
    indices0 = 2*np.arange(num)
    indices1 = indices0+1
    # acc = (preds[indices].argmax(dim=1) == labels[indices]).float().mean()
    err0= (preds[indices0].argmax(dim=1).detach().numpy() != labels[indices0].detach().numpy()).astype(int)
    err1= (preds[indices1].argmax(dim=1).detach().numpy() != labels[indices1].detach().numpy()).astype(int)
    return err0, err1


def get_fcmi_results_for_fixed_z(n, lr, epoch, seed, args):
    train_accs = []
    val_accs = []
    preds = []
    masks = []
    L0, L1=[], []
    all_examples = None  # will be needed after this loop to dump some extra information
    for S_seed in range(args.n_S_seeds):
        #S_seed = S_seed + 27 #DELETE THIS!!!
        dir_name = f'n={n},lr={lr},seed={seed},S_seed={S_seed}'
        dir_path = os.path.join(args.results_dir, args.exp_name, dir_name)
        if not os.path.exists(dir_path):
            print(f"Did not find results for {dir_name}")
            continue

        with open(os.path.join(dir_path, 'saved_data.pkl'), 'rb') as f:
            saved_data = pickle.load(f)

        model = utils.load(path=os.path.join(dir_path, 'checkpoints', f'epoch{epoch - 1}.mdl'),
                           methods=methods, device=args.device)

        print(list(saved_data.keys()))
        if 'all_examples_wo_data_aug' in saved_data:
            all_examples = saved_data['all_examples_wo_data_aug']
        else:
            all_examples = saved_data['all_examples']

        cur_preds = utils.apply_on_dataset(model=model, dataset=all_examples,
                                           batch_size=args.batch_size)['pred']
        cur_mask = saved_data['mask']
        cur_train_acc = compute_acc(preds=cur_preds, mask=cur_mask, dataset=all_examples)
        cur_val_acc = compute_acc(preds=cur_preds, mask=1-cur_mask, dataset=all_examples)
        print(cur_train_acc, cur_val_acc)
        Lposs, Lnegs = compute_error(preds=cur_preds, num=n, dataset=all_examples)
        L0.append(Lposs)
        L1.append(Lnegs)
        preds.append(cur_preds)
        masks.append(cur_mask)
        train_accs.append(cur_train_acc)
        val_accs.append(cur_val_acc)

    # delta_bound, pos_bound, neg_bound, opt_bound, sub_bound, single_ip_bound, var_bound, sharp_bound, channel_capacity=estimate_ecmi_bound_classification(masks=masks, L0=L0, L1=L1,
    #                                                                             num_examples=n, train_acc = train_accs)
    newdelta_bound, fast_bound, hd_bound, hd_bound1, hd_bound2, jsd_bound, kltv_bound, delta_bound, sharp_bound, channel_capacity, MI, Chi2, HD, JSD, ZCP = estimate_ecmi_bound_classification(
        masks=masks, L0=L0, L1=L1,
        num_examples=n, train_acc=train_accs)
    fcmi_bound = estimate_fcmi_bound_classification(masks=masks, preds=preds,
                                                    num_examples=n, num_classes=args.num_classes)
    kl_bound = estimate_kl_bound_classification(masks=masks, preds=preds,
                                                    num_examples=n, num_classes=args.num_classes,
                                                    train_acc = np.mean(train_accs))
    # interp_bound = estimate_interp_bound_classification(masks=masks, preds=preds,
    #                                                 num_examples=n, num_classes=args.num_classes,
    #                                                 train_acc = np.mean(train_accs))
    lg_bound = estimate_lg_bound_classification(masks=masks, preds=preds,
                                                    num_examples=n, num_classes=args.num_classes,
                                                    train_acc = np.mean(train_accs))

    # some extra for understanding why the f-cmi bound is high in the beginning (fcmi-mnist-4vs9-CNN-LD)
    if args.exp_name == 'fcmi-mnist-4vs9-CNN-LD' and epoch == 4 and seed == 0:
        extra_data = {
            'all_examples': all_examples,
            'masks': masks,
            'preds': preds,
            'num_examples': n,
            'num_classes': args.num_classes
        }
        with open('results/fcmi-mnist-4vs9-CNN-LD/extra_data.pkl', 'wb') as f:
            pickle.dump(extra_data, f)
    return {
        'exp_train_acc': np.mean(train_accs),
        'exp_val_acc': np.mean(val_accs),
        'exp_gap': np.mean(train_accs) - np.mean(val_accs),
        'fcmi_bound': fcmi_bound,
        'kl_bound': kl_bound,
        'newdelta_bound': newdelta_bound,
        'fast_bound': fast_bound,
        'hd_bound': hd_bound,
        'hd_bound1': hd_bound1,
        'hd_bound2': hd_bound2,
        'jsd_bound': jsd_bound,
        'kltv_bound': kltv_bound,
        'delta_bound': delta_bound,
        'sharp_bound': sharp_bound,
        'channel_capacity': channel_capacity,
        'MI': MI,
        'Chi2': Chi2,
        'HD': HD,
        'JSD': JSD,
        'ZCP': ZCP,
        'lg_bound': lg_bound
    }
    # return {
    #     'exp_train_acc': np.mean(train_accs),
    #     'exp_val_acc': np.mean(val_accs),
    #     'exp_gap': np.mean(train_accs) - np.mean(val_accs),
    #     'fcmi_bound': fcmi_bound,
    #     'kl_bound': kl_bound,
    #     'interp_bound': interp_bound,
    #     'delta_bound': delta_bound,
    #     'pos_bound': pos_bound,
    #     'neg_bound': neg_bound,
    #     'opt_bound': opt_bound,
    #     'sub_bound': sub_bound,
    #     'single_ip_bound': single_ip_bound,
    #     'var_bound': var_bound,
    #     'sharp_bound': sharp_bound,
    #     'channel_capacity':channel_capacity,
    #     'lg_bound': lg_bound
    # }


def get_fcmi_results_for_fixed_model(n, lr, epoch, args):
    results = []
    for seed in range(args.n_seeds):
        #seed = seed + 1 #DELETE THIS!!
        cur = get_fcmi_results_for_fixed_z(n=n, lr=lr, epoch=epoch, seed=seed, args=args)
        results.append(cur)
    return results


def get_sgld_results_for_fixed_model(n, lr, epoch, args):
    results = []

    for seed in range(args.n_seeds):
        #seed = seed + 1 #DELETE THIS!!
        for S_seed in range(args.n_S_seeds):
            #S_seed = S_seed + 27 #DELETE THIS!!!!
            if S_seed >= 4:
                continue  # these guys didn't track gradient variance to save time

            dir_name = f'n={n},lr={lr},seed={seed},S_seed={S_seed}'
            dir_path = os.path.join(args.results_dir, args.exp_name, dir_name)
            if not os.path.exists(dir_path):
                print(f"Did not find results for {dir_name}")
                continue

            with open(os.path.join(dir_path, 'saved_data.pkl'), 'rb') as f:
                saved_data = pickle.load(f)

            model = utils.load(path=os.path.join(dir_path, 'checkpoints', f'epoch{epoch - 1}.mdl'),
                               methods=methods, device=args.device)

            if 'all_examples_wo_data_aug' in saved_data:
                all_examples = saved_data['all_examples_wo_data_aug']
            else:
                all_examples = saved_data['all_examples']

            cur_preds = utils.apply_on_dataset(model=model, dataset=all_examples,
                                               batch_size=args.batch_size)['pred']
            cur_mask = saved_data['mask']
            cur_train_acc = compute_acc(preds=cur_preds, mask=cur_mask, dataset=all_examples)
            cur_val_acc = compute_acc(preds=cur_preds, mask=1-cur_mask, dataset=all_examples)

            cur_result = {}
            cur_result['train_acc'] = cur_train_acc
            cur_result['val_acc'] = cur_val_acc
            cur_result['gap'] = cur_val_acc - cur_train_acc

            sgld_bound = estimate_sgld_bound(n=n, batch_size=args.batch_size,
                                             model=model)
            cur_result['sgld_bound'] = sgld_bound
            results.append(cur_result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.set_defaults(parse=True)
    args = parser.parse_args()
    print(args)

    if args.exp_name in ["fcmi-mnist-4vs9-CNN", "fcmi-mnist-4vs9-CNN-deterministic", "fcmi-mnist-4vs9-CNN-noisy", "fcmi-mnist-4vs9-CNN-halfnoisy"]:
        args.n_seeds = 5
        args.n_S_seeds = 30
        args.ns = [75, 250, 1000, 4000]
        args.epochs = [200]#args.epochs = np.arange(1, 11) * 20
        args.num_classes = 2
    elif args.exp_name in ["fcmi-mnist-4vs9-CNN-lr"]:
        print("hiiya")
        args.n_seeds = 5
        args.n_S_seeds = 30
        args.ns = [4000]#[75]
        args.lrs = [0.1, 1e-3, 1e-5]
        args.epochs = [200]#args.epochs = np.arange(1, 11) * 20
        args.num_classes = 2
    elif args.exp_name in ["fcmi-mnist-4vs9-CNN-width","fcmi-mnist-4vs9-CNN-depth"]:
        print("hi")
        args.n_seeds = 5
        args.n_S_seeds = 30
        args.ns = [4000]#[75]
        args.lrs = [0.0, 1.0, 2.0]
        args.epochs = [200]#args.epochs = np.arange(1, 11) * 20
        args.num_classes = 2
    elif args.exp_name == 'fcmi-mnist-4vs9-wide-CNN-deterministic':
        args.n_seeds = 5
        args.n_S_seeds = 30
        args.ns = [75, 250, 1000, 4000]
        args.epochs = [200]
        args.num_classes = 2
    elif args.exp_name == 'fcmi-mnist-4vs9-CNN-LD':
        args.n_seeds = 5
        args.n_S_seeds = 30
        args.ns = [4000]
        args.epochs = np.arange(1, 11) * 4
        args.num_classes = 2
        args.batch_size = 100
    elif args.exp_name == 'cifar10-pretrained-resnet50':
        args.n_seeds = 2
        args.n_S_seeds = 40
        args.ns = [1000, 5000, 20000]
        args.epochs = [40]
        args.num_classes = 10
    elif args.exp_name == 'cifar10-pretrained-resnet50-LD':
        args.n_seeds = 1
        args.n_S_seeds = 40
        args.ns = [20000]
        args.epochs = np.arange(1, 9) * 2
        args.num_classes = 10
        args.batch_size = 64
    else:
        raise ValueError(f"Unexpected exp_name: {args.exp_name}")

    # parse quantities needed for the f-CMI bound
    results = NestedDict()  # indexing with n, epoch
    if args.exp_name in ["fcmi-mnist-4vs9-CNN-lr", "fcmi-mnist-4vs9-CNN-width", "fcmi-mnist-4vs9-CNN-depth"]:
        n = 4000
        for lr in tqdm(args.lrs):
            for epoch in tqdm(args.epochs, leave=False):
                results[lr][epoch] = get_fcmi_results_for_fixed_model(n=n, lr=lr, epoch=epoch, args=args)
        results_file_path = os.path.join(args.results_dir, args.exp_name, 'results.pkl')
        with open(results_file_path, 'wb') as f:
            pickle.dump(results, f)

    elif args.exp_name in ['cifar10-pretrained-resnet50']:
        for n in tqdm(args.ns):
            for epoch in tqdm(args.epochs, leave=False):
                results[n][epoch] = get_fcmi_results_for_fixed_model(n=n, lr=1e-2, epoch=epoch, args=args)
        results_file_path = os.path.join(args.results_dir, args.exp_name, 'results.pkl')
        with open(results_file_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        for n in tqdm(args.ns):
            for epoch in tqdm(args.epochs, leave=False):
                results[n][epoch] = get_fcmi_results_for_fixed_model(n=n, lr=1e-3, epoch=epoch, args=args)
        results_file_path = os.path.join(args.results_dir, args.exp_name, 'results.pkl')
        with open(results_file_path, 'wb') as f:
            pickle.dump(results, f)

    # parse the quantities needed for the Negrea et al. SGLD bound
    if args.exp_name in ['fcmi-mnist-4vs9-CNN-LD', 'cifar10-pretrained-resnet50-LD']:
        sgld_results = NestedDict()  # indexing with n, epoch
        for n in tqdm(args.ns):
            for epoch in tqdm(args.epochs, leave=False):
                sgld_results[n][epoch] = get_sgld_results_for_fixed_model(n=n, lr = 1e-3, epoch=epoch, args=args)
        results_file_path = os.path.join(args.results_dir, args.exp_name, 'sgld_results.pkl')
        with open(results_file_path, 'wb') as f:
            pickle.dump(sgld_results, f)


if __name__ == '__main__':
    main()
