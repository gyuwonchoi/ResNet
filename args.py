import argparse

parser = argparse.ArgumentParser(
        description="Pytorch Implementation for ResNet")
parser.add_argument('--batch_size', type=int, default=196)
parser.add_argument('--layer', type=int, default=20)
parser.add_argument('--epoch', type=int, default=64000)
parser.add_argument('--dataset', type=str, default ='cifar10')
parser.add_argument('--lr', type=float, default = 0.1)
