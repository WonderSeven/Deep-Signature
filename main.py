import os
import sys
import argparse
from PIL import ImageFile

sys.dont_write_bytecode = True
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser('Gene regulatory Dynamic Case')
    # Dataset setup
    parser.add_argument('--data_root', type=str, default='/home/qtx/Datasets/Gene_regulatory/Gene_N400_T2_Tick500_50')
    parser.add_argument('--data_name', type=str, default='GeneRegulation')
    parser.add_argument('--data_reg', type=str, default='gene')
    parser.add_argument('--atom_type', type=str, choices=['all', 'backbone', 'alpha_carbon', 'residue'], default='all')
    parser.add_argument('--traj_nums', type=int, default=50, help='The number of trajectories')
    parser.add_argument('--groups', type=int, default=5, help='The number of groups')
    parser.add_argument('--folds', type=int, default=5, help='The number of groups')

    # Algorithm Components
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--algorithm', type=str, default='UniversalSignature')
    parser.add_argument('--local_mode', type=str, default='head_and_tail')
    parser.add_argument('--input_dim', type=int, default=1, help='The dims fo hidden variables')
    parser.add_argument('--hidden_dim', type=int, default=10, help='The dims fo hidden variables')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability)')
    parser.add_argument('--num_clusters', type=int, default=25, help='Number of nodes')

    # Optimization params
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for ProtoTransfer')
    parser.add_argument('--test_epoch', type=int, default=-2, help='Epoch when model test')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of workers for data loaders')

    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate') # 5e-6
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).') # 1e-6
    parser.add_argument('--loss_func', type=str, choices=['cross_entropy', 'mse', 'f1', 'bce'], default='f1')
    parser.add_argument('--patience', type=int, default=-1, help='Patience until early stopping. -1 means no early stopping')
    # Environment params
    parser.add_argument('--save_path', type=str, default='./logs/Test', help='Save path')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--record', action='store_true', help='Whether to record the model training procedure')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    print('GPUs id:{}'.format(args.gpu_ids))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    from engine import trainer
    trainer = trainer.Trainer(args)
    trainer.train()
