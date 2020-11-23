import argparse
import easydict


def load_args():
    parser = argparse.ArgumentParser('BERT')

    # dataset
    parser.add_argument('--nsp_ratio', type=float, default=0.5)
    parser.add_argument('--mlm_ratio', type=float, default=0.15)
    parser.add_argument('--max_len', type=int, default=32)

    # data loader
    parser.add_argument('--base_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)

    # model
    parser.add_argument('--N', type=int, default=12, help=[12, 24])
    parser.add_argument('--embedding_dim', type=int, default=768, help=[768, 1024])
    parser.add_argument('--heads', type=int, default=12, help=[12, 16])

    # training
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=150)

    # down-stream task
    parser.add_argument('--task', type=bool, default=None)
    parser.add_argument('--checkpoints', type=str, default=None)

    args = parser.parse_args()

    return args


def load_easydict():
    args = easydict.EasyDict({
        "nsp_ratio": 0.5,
        "mlm_ratio": 0.15,
        "max_len": 32,
        "base_dir": './data',
        "batch_size": 64,
        "num_workers": 4,
        "N": 12,
        "embedding_dim": 768,
        "heads": 12,
        "lr": 1e-4,
        "weight_decay": 0.01
    })

    return args