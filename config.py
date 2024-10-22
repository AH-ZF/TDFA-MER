import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='TDFA', type=str,
                    help='deep networks to be trained')

parser.add_argument('--size', default=10,
                    help='the size of blocks')
parser.add_argument('--is_split', default=False,
                    help='whether to use spliting data (default: False)')
parser.add_argument('--isallfeature', default=True,
                    help='hyper-patameter_lambda for ISDA')
parser.add_argument('--isallfeaStruct', default=23, type=int,
                    help='hyper-patameter_lambda for ISDA')
parser.add_argument('--ram', default=False, type=bool,
                    help='  ')

# parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
#                     help='path to save checkpoint (default: checkpoint)')

parser.add_argument('--save_path', default='result/TDFA', type=str,
                    help='the save path of results')
# parser.add_argument('--protocol', default='loso', type=str,
#                     help='loso or lovo or randoms')

# train phase*******************************************************************************************

parser.add_argument('--chooseNormal', default=3, type=int,
                    help='0,1,2,3,4: CASMEII,SAMM,SMIC,Composite3,Composite5')
parser.add_argument('--gpunum', default='cuda:0',
                    help='whether to use send email to zf(False or True)')
parser.add_argument('--ispretrained', default=True,
                    help='whether to use   to zf(False or True)')
parser.add_argument('--is_augment', default=False,
                    help='whether to use data augment(default: False)')
parser.add_argument('--isXavier', default=False,
                    help='whether to use data augment(default: False)')
parser.add_argument('--issavewbinit', default=True,
                    help='whether to use data augment(default: False)')
parser.add_argument('--Epoch', default=100, type=int,
                    help='  ')
parser.add_argument('--train_shuffle', default=False, type=bool,
                    help='  ')
parser.add_argument('--trainbatch_size', default=64, type=int,
                    help='  ')
parser.add_argument('--changelr', default=12, type=int,
                    help='-1:paper; 1:lr1; 2:lr2, 3:lr3==0.01, 4:lr4, 5:lr5, 6：lr2_n,7:lr3_new,8:lr8,9:lr9')
# *******************************************************************************************************


parser.add_argument('--testbatch_size', default=4, type=int,
                    help='  ')
parser.add_argument('--initial_learning_rate', default=0.01, type=float,
                    help='  ')
parser.add_argument('--changing_lr', default=[80, 120],
                    help='  ')
parser.add_argument('--lr_decay_rate', default=0.1, type=float,
                    help='  ')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='  ')
parser.add_argument('--nesterov', default=True, type=bool,
                    help='  ')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='  ')
parser.add_argument('--lr_strategy', default=-1, type=int,
                    help='  ')
# Breakpoint training
parser.add_argument('--savecheckpoint', default=False, type=bool,
                    help=' False or True ')
parser.add_argument('--loadcheckpoint', default=False, type=bool,
                    help=' False or True ')

parser.set_defaults(cos_lr=False)
args = parser.parse_args()
