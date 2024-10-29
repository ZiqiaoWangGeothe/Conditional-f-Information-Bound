import argparse

from nnlib.nnlib.data_utils.base import register_parser as register_fn


local_functions = {}  # storage for registering the functions below


#######################################################################################
#
#     MNIST 4 vs 9, CNN
#
# exp_name: fcmi-mnist-4vs9-CNN
#######################################################################################

@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN')
def foo(deterministic=False, **kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = "fcmi-mnist-4vs9-CNN"
    if deterministic:
        exp_name = exp_name + '-deterministic'
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};"
                print(command)
                # exec(command)


@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-lr')
def foo(deterministic=False, **kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = "fcmi-mnist-4vs9-CNN-lr"
    if deterministic:
        exp_name = exp_name + '-deterministic'
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"!python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "

    n_seeds = 5
    n_S_seeds = 30
    n = 4000
    lrs = [1e-1, 1e-3, 1e-5]

    for lr in lrs:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --lr {lr} --seed {seed} --S_seed {S_seed};"
                print(command)

@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-width')
def foo(deterministic=False, **kwargs):
    configs = ['configs/binary-mnist-4layer-CNN.json', 
              'configs/binary-mnist-4layer-midwide-CNN.json',
              'configs/binary-mnist-4layer-wide-CNN.json']
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = "fcmi-mnist-4vs9-CNN-width"
    if deterministic:
        exp_name = exp_name + '-deterministic'
    dataset = 'mnist'
    which_labels = '4 9'

    

    n_seeds = 5
    n_S_seeds = 30
    n = 4000
    lrs = [0, 1, 2]

    for lr in lrs:
        config_file = configs[lr]
        command_prefix = f"!python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --lr {lr} --seed {seed} --S_seed {S_seed};"
                print(command)


@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-depth')
def foo(deterministic=False, **kwargs):
    configs = ['configs/binary-mnist-4layer-lowdeep-CNN.json',
              'configs/binary-mnist-4layer-CNN.json',
              'configs/binary-mnist-4layer-deep-CNN.json']
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = "fcmi-mnist-4vs9-CNN-depth"
    if deterministic:
        exp_name = exp_name + '-deterministic'
    dataset = 'mnist'
    which_labels = '4 9'

    

    n_seeds = 5
    n_S_seeds = 30
    n = 4000
    lrs = [0, 1, 2]

    for lr in lrs:
        config_file = configs[lr]
        command_prefix = f"!python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --lr {lr} --seed {seed} --S_seed {S_seed};"
                print(command)


@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-noisy')
def foo(deterministic=False, **kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = "fcmi-mnist-4vs9-CNN-noisy"
    if deterministic:
        exp_name = exp_name + '-deterministic'
    dataset = 'uniform-noise-mnist'
    which_labels = '4 9'

    # command_prefix = f"!python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
    #                  f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
    #                  f"--which_labels {which_labels} --deterministic "

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};"
                print(command)

@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-halfnoisy')
def foo(deterministic=False, **kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 20
    exp_name = "fcmi-mnist-4vs9-CNN-halfnoisy"
    if deterministic:
        exp_name = exp_name + '-deterministic'
    dataset = 'uniform-noise-mnist'
    which_labels = '4 9'

    # command_prefix = f"!python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
    #                  f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
    #                  f"--which_labels {which_labels} --deterministic "

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};"
                print(command)

@register_fn(local_functions, 'fcmi-mnist-4vs9-wide-CNN-deterministic')
def foo(**kwargs):
    config_file = 'configs/binary-mnist-4layer-wide-CNN.json'
    batch_size = 128
    n_epochs = 200
    save_iter = 200
    exp_name = "fcmi-mnist-4vs9-wide-CNN-deterministic"
    dataset = 'mnist'
    which_labels = '4 9'

    command_prefix = f"!python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} --deterministic "

    n_seeds = 5
    n_S_seeds = 30
    ns = [75, 250, 1000, 4000]

    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};"
                print(command)


@register_fn(local_functions, 'fcmi-mnist-4vs9-CNN-LD')
def foo(**kwargs):
    config_file = 'configs/binary-mnist-4layer-CNN.json'
    batch_size = 100
    n_epochs = 40
    save_iter = 4
    exp_name = "fcmi-mnist-4vs9-CNN-LD"
    dataset = 'mnist'
    which_labels = '4 9'
    ld_lr = 0.004
    ld_beta = 10.0

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"--which_labels {which_labels} -m LangevinDynamics --ld_lr {ld_lr} "\
                     f"--ld_beta {ld_beta} "

    n_seeds = 5
    n_S_seeds = 30
    ns = [4000]

    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed}"
                if S_seed < 4:  # producing in total 20 runs with this flag
                    command += " --ld_track_grad_variance"
                command += ";"
                print(command)


@register_fn(local_functions, 'cifar10-pretrained-resnet50')
def foo(**kwargs):
    config_file = 'configs/pretrained-resnet50-cifar10.json'
    batch_size = 64
    n_epochs = 40
    save_iter = n_epochs
    exp_name = "cifar10-pretrained-resnet50"
    dataset = 'cifar10'
    optimizer = 'sgd'
    lr = 0.01
    momentum = 0.9

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"-m StandardClassifier -A --resize_to_imagenet --optimizer {optimizer} "\
                     f"--lr {lr} --momentum {momentum} "

    n_seeds = 2
    n_S_seeds = 40
    ns = [1000, 5000, 20000]

    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed};"
                print(command)


@register_fn(local_functions, 'cifar10-pretrained-resnet50-LD')
def foo(**kwargs):
    config_file = 'configs/pretrained-resnet50-cifar10.json'
    batch_size = 64
    n_epochs = 16
    save_iter = 2
    exp_name = "cifar10-pretrained-resnet50-LD"
    dataset = 'cifar10'
    ld_lr = 0.01
    ld_beta = 10.0

    command_prefix = f"python -um scripts.fcmi_train_classifier -c {config_file} -d cuda -b {batch_size} " \
                     f"-e {n_epochs} -s {save_iter} -v 10000 --exp_name {exp_name} -D {dataset} " \
                     f"-m LangevinDynamics -A --resize_to_imagenet  "\
                     f"--ld_lr {ld_lr} --ld_beta {ld_beta} "

    n_seeds = 1
    n_S_seeds = 40
    ns = [20000]

    for n in ns:
        for seed in range(n_seeds):
            for S_seed in range(n_S_seeds):
                command = command_prefix + f"--n {n} --seed {seed} --S_seed {S_seed}"
                if S_seed < 4:  # producing in total 4 runs with this flag
                    command += " --ld_track_grad_variance --ld_track_every_iter 10"
                command += ";"
                print(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', '-E', type=str, nargs='+', required=True)
    parser.add_argument('--deterministic', action='store_true', dest='deterministic')
    args = parser.parse_args()

    for exp_name in args.exp_names:
        assert exp_name in local_functions
        local_functions[exp_name](**vars(args))


if __name__ == '__main__':
    main()
