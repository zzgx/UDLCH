import argparse

parser = argparse.ArgumentParser(description='implementation')

parser.add_argument("--data_name", type=str, default="mirflickr25k_fea",
                    help="data name")
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--log_name', type=str, default='all_log')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='all_checkpoints')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11', help='model architecture: ' + ' | '.join(['ResNet', 'VGG']) + ' (default: vgg11)')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=1e-6)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=256)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--task_epochs', type=int, default=5)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--num_hiden_layers', default=[3, 2], nargs='+', help='<Required> Number of hiden lyaers')
parser.add_argument('--ls', type=str, default='linear', help='lr scheduler')
parser.add_argument('--bit', type=int, default=32, help='output shape')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--momentum', type=float, default=0.4)
parser.add_argument('--K', type=int, default=4096)
parser.add_argument('--T', type=float, default=.9)
parser.add_argument('--shift', type=float, default=1)
parser.add_argument('--margin', type=float, default=.2)
parser.add_argument('--warmup_epoch', type=int, default=1)
parser.add_argument('--num_tasks', type=float, default=10)
parser.add_argument('--buffer_size', type=float, default=500)
parser.add_argument('--category_split_ratio', type=str, default="(23, 1)")
parser.add_argument('--der_a', type=float, default=20.)
parser.add_argument('--der_b', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=1.)

args = parser.parse_args()


try:
    args.category_split_ratio = eval(args.category_split_ratio)
    if not isinstance(args.category_split_ratio, tuple) or len(args.category_split_ratio) != 2:
        raise ValueError("category_split_ratio must be a tuple with two elements")
except:
    raise ValueError("Invalid format for category_split_ratio. Use '(A, B)' format.")
args.num_hiden_layers = [int(i) for i in args.num_hiden_layers]
args.num_tasks = int(args.num_tasks)
args.buffer_size = int(args.buffer_size)


