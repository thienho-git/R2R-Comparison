
# %%
import torch
import torch
print(torch.cuda.is_available())
print(torch.cuda.memory_allocated()/1e9, "GB allocated")
print(torch.cuda.memory_reserved()/1e9, "GB reserved")
# %%
import os, sys, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'vamtoolbox_devR2R')) # devR2R branch of vamtoolbox, not the venv's main branch
import vamtoolbox as vam
import numpy as np
np.product = np.prod # aliases removed in NumPy 2, still used by vamtoolbox devR2R
np.NaN = np.nan
import logging
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
import os
import sys
import PIL.Image
import time
import skimage.draw
from scipy.io import savemat

logging.basicConfig() #initialize the logging module
logger = logging.getLogger('vamtoolbox') #Set the logging level in the whole vamtoolbox
logger.setLevel(logging.DEBUG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
logger.info('Computing on device: {}'.format(device))
logger.info(torch.cuda.device_count())

class configr2r:
    def __init__(self,name,alpha_mat,n_sur,n_mat,tau,r_i,l_max,deg_inc,gradient=1.0,hours_set=0,p_cubic=2,shrink_z=1):
        
        self.name = name
        self.alpha_mat = alpha_mat
        self.n_sur = n_sur
        self.n_mat = n_mat
        self.tau = tau # thickness in cm
        self.r_i = r_i # roller radius
        self.r_o = self.r_i + self.tau 

        self.l_max = l_max
        self.gradient = gradient

        self.hours_set = hours_set

        self.deg_inc = deg_inc
        self.f_rho = 1/(0.00108*1.33) # 10.8um is the DMD native pixel size; 1.33 is the megnification factor
        self.f_l = 1/(self.deg_inc*np.pi/180*self.r_i) # tangential frequency

        self.anisotropy_factor = self.f_rho/self.f_l
        
        self.N_l = int(self.f_l*self.l_max)
        self.N_rho = int(self.f_rho*(self.r_o-self.r_i))
        self.N_z = int(self.f_rho*self.tau)

        self.p_cubic = p_cubic
        self.shrink_z = shrink_z
    
    def __repr__(self,):
        '''Returns config string with name and all parameters'''
        # return (f'{self.name}_alpha_mat={self.alpha_mat}_n_sur={self.n_sur}_n_mat={self.n_mat}_r_i={self.r_i}_tau={self.tau}_l_max={self.l_max}_gradient={self.gradient}_deg_inc={self.deg_inc}_hrs_set={self.hours_set}')
        # return (f'{self.name}_alpha_mat={self.alpha_mat}_n_sur={self.n_sur}_n_mat={self.n_mat}_l_max={self.l_max}_deg_inc={self.deg_inc}')
        """Returns config string with name and all parameters"""
        parts = [
            f"{self.name}",
            f"alpha_mat={self.alpha_mat}",
            f"n_sur={self.n_sur}",
            f"n_mat={self.n_mat}",
            f"r_i={self.r_i}",
            f"tau={self.tau}",
            f"l_max={self.l_max}",
            f"gradient={self.gradient}",
            f"deg_inc={self.deg_inc}",
            f"hrs_set={self.hours_set}",
        ]

        if hasattr(self, "p_cubic") and (self.p_cubic is not None):
            parts.append(f"p_cubic={self.p_cubic}")
        if hasattr(self, "shrink_z") and (self.shrink_z is not None):
            parts.append(f"shrink_z={self.shrink_z}")

        # Option B (stricter): only include for certain types
        # lattice_types = {"cubic-beams", "cubic-skel"}
        # cur_type = getattr(self, "tpms_type", getattr(self, "type", None))
        # if cur_type in lattice_types:
        #     parts.append(f"p_cubic={getattr(self, 'p_cubic', None)}")

        return "_".join(parts)


    def printFull(self,):
        print(f'{self.name}')
        print(f'alpha_mat = {self.alpha_mat} 1/cm')
        print(f'n_sur = {self.n_sur} n_mat = {self.n_mat}') 
        print(f'tau = {self.tau} cm')
        print(f'r_i = {self.r_i} cm')
        print(f'f_rho = {self.f_rho} voxels/cm f_l = {self.f_l} voxels/cm')
        print(f'deg_inc = {self.deg_inc} °')
        print(f'l_max = {self.l_max} cm')
        print(f'hours_set = {self.hours_set} hr')
        print(f'N_rho = {self.N_rho} pixels')
        print(f'N_l = {self.N_l} pixels')
    def getFolderName(self):
        # Construct a simple filename-safe version of the config
        return f"{self.name}_tau{self.tau}_ri{self.r_i}_lmax{self.l_max}".replace(" ", "_").replace(":", "-")

    def matrixStr(self,):
        # Replace or remove problematic characters
        '''Returns config string with only propagation relevant parameters'''
        return (f'alpha_mat={self.alpha_mat}_n_sur={self.n_sur}_n_mat={self.n_mat}_r_i={self.r_i}_tau={self.tau}_deg_inc={self.deg_inc}')
        # return (f'alpha_mat={self.alpha_mat}_n_sur={self.n_sur}_n_mat={self.n_mat}_reflection_r_i={self.r_i}_tau={self.tau}_deg_inc={self.deg_inc}')
        # print(f"Spatial sampling rate rho = {spatial_sampling_rate_rho}")
        # print(f"Spatial sampling rate L = {spatial_sampling_rate_l}")
        # print(f"Anistropy factor = {anisotropy_factor}")
        # print(f"Target shape = {N_rho},{N_l},{N_z}")

parser = argparse.ArgumentParser()
parser.add_argument("geometry", nargs="?", default="tpms_diamond_skel0", help="one of the cases below")
parser.add_argument("--target-file", default=None, help=".npy target (N_rho, N_l, N_z) used instead of the case's own target")
parser.add_argument("--output-dir", default=None, help="results folder (default: outputs/reference_standalone/<config>)")
parser.add_argument("--matrix-dir", default=os.path.join(PROJECT_DIR, "outputs", "matrices"), help="folder of the matrix_sino2uw_*.pt files")
args, _ = parser.parse_known_args()
geometry = args.geometry
output_root = args.output_dir or os.path.join(PROJECT_DIR, "outputs", "reference_standalone")
os.makedirs(output_root, exist_ok=True)
match geometry:
    case "TPMS carved Quote - Hocus Pocus - 5cm - seg1":
        config = configr2r(geometry,1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5)
        a = np.load(args.target_file) if args.target_file else loadmat(os.path.join(output_root, 'TPMS letter target', 'TPMS carved Hocus Pocus 208_by_1400_by_1040.mat'))["target"]
        a = a[:, 0:753, :]
        a = a[:, :, 0:208]
        print(a.shape)
    case "tpms_bcc_skel15":
        unit_cell_size = 0.3
        config = configr2r(geometry, 2.7, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.01])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=unit_cell_size,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel14":
        unit_cell_size = 0.3
        config = configr2r(geometry, 2.7, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.025])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=unit_cell_size,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel13":
        unit_cell_size = 0.3
        config = configr2r(geometry, 2.8, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.025])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=unit_cell_size,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel12":
        unit_cell_size = 0.3
        config = configr2r(geometry, 2.6, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.025])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=unit_cell_size,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel11":
        unit_cell_size = 0.15
        config = configr2r(geometry, 3, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=unit_cell_size,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel10":
        unit_cell_size = 0.2
        config = configr2r(geometry, 3, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=unit_cell_size,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()

    case "tpms_bcc_skel9":
        config = configr2r(geometry, 3, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()

    case "tpms_bcc_skel8":
        config = configr2r(geometry, 3.2, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel7":
        config = configr2r(geometry, 2.8, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel6":
        config = configr2r(geometry, 2.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel0":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.075])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel1":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel2":
        config = configr2r(geometry, 1.8, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel3":
        config = configr2r(geometry, 1.6, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel4":
        config = configr2r(geometry, 2.2, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "tpms_bcc_skel5":
        config = configr2r(geometry, 2.5, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.05])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='bcc-skel')
        os.chdir(output_root)
        savemat('target.mat', {'target': a})

        plt.imshow(a[:, 370, :])
        plt.show()
    case "resolution_charcaterization_v1":
        config = configr2r(geometry, 2.2, 1.33, 1.49, 0.3, 0.76, 5, 0.5)
        data = {'target': np.load(args.target_file)} if args.target_file else loadmat('G:/Shared drives/taylorlab/CAL Projects/R2RCAL/paper revision/target_resolution_target_matrix_v1.mat')
        a = data['target']
        print(a.shape)
    case "tpms_diamond_skel0":
        config = configr2r(geometry,1.2, 1.33, 1.49, 0.3, 0.76, 5.0, 0.5, [0.5])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='diamond-skel')
    case "tpms_diamond_skel6":
        config = configr2r(geometry, 2.5, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.5])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='diamond-skel')
    case "tpms_diamond_skel7":
        config = configr2r(geometry, 2.6, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.5])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='diamond-skel')
    case "tpms_diamond_skel8":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.5])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='diamond-skel')
    case "tpms_diamond_skel9":
        config = configr2r(geometry, 1.3, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.5])
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='diamond-skel')
    case "tpms_cubic_ske0":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.5], p_cubic=2)
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='cubic-skel', p_cubic=config.p_cubic)
    case "tpms_cubic_ske1":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.5], p_cubic=2, shrink_z=1.5)
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='cubic-skel', p_cubic=config.p_cubic, shrink_z=config.shrink_z)
        os.chdir(output_root)
        savemat('target.mat', {'target': a})
    case "tpms_cubic_ske2":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.5], p_cubic=2, shrink_z=2)
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='cubic-skel', p_cubic=config.p_cubic, shrink_z=config.shrink_z)
        os.chdir(output_root)
        savemat('target.mat', {'target': a})
    case "tpms_cubic_ske3":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.3], p_cubic=2)
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='cubic-skel', p_cubic=config.p_cubic, shrink_z=config.shrink_z)
        os.chdir(output_root)
        savemat('target.mat', {'target': a})
    case "tpms_cubic_ske4":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.3], p_cubic=2, shrink_z=1.5)
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='cubic-skel', p_cubic=config.p_cubic, shrink_z=config.shrink_z)
        os.chdir(output_root)
        savemat('target.mat', {'target': a})
    case "tpms_cubic_ske5":
        config = configr2r(geometry, 1.4, 1.33, 1.49, 0.3, 0.76, 5, 0.5, [0.8], p_cubic=2)
        a = vam.util.buildtargetsR2R.tpms((config.N_rho, config.N_l, config.N_z),f_rho=config.f_rho,f_l=config.f_l,unit_cell_size=config.tau,t=config.gradient,anisotropy_factor=config.anisotropy_factor,type='cubic-skel', p_cubic=config.p_cubic)
        os.chdir(output_root)
        savemat('target.mat', {'target': a})
# %%        
if args.target_file:
    a = np.load(args.target_file) # target given on the command line instead of the case's own target
directory = args.output_dir or os.path.join(output_root, f"{config}")

if not os.path.exists(directory):
    os.makedirs(directory)

target_geo = vam.geometry.TargetGeometryR2R(target=a,r_i=config.r_i,tau=config.tau,spatial_sampling_rate_rho=config.f_rho,spatial_sampling_rate_l=config.f_l)
coord_vec = target_geo.constructCoordVec()
# %%
target_geo.show()
np.save(os.path.join(directory,'target.npy'),target_geo.array)
# %%
fig, axs = plt.subplots(1,1)
im_recon = axs.imshow(target_geo.array[:,:,target_geo.array.shape[2]//2],extent=[0,target_geo.nL_total/target_geo.transform.spatial_sampling_rate_l,target_geo.transform.r_o,target_geo.transform.r_i],cmap='gray')
im_recon.set_clim(0,1)
axs.set_xlabel(r'$L$ [cm]')
axs.set_ylabel(r'$\rho$ [cm]')
vam.util.matplotlib.addColorbar(im_recon)
fig.tight_layout()
fig.savefig(os.path.join(directory,'target.png'),dpi=600)
plt.show()


attenuation_model = vam.medium.AttenuationModel(coord_vec, type = 'interpolation', form='homogeneous_r2r', alpha_internal=config.alpha_mat, rho_grid=target_geo.transform.rho_g_w, r_mid=target_geo.transform.r_mid, tau=target_geo.transform.tau)
absorption_model = vam.medium.AttenuationModel(coord_vec, type = 'interpolation', form='homogeneous_r2r', alpha_internal=config.alpha_mat, rho_grid=target_geo.transform.rho_g_w, r_mid=target_geo.transform.r_mid, tau=target_geo.transform.tau)
# attenuation_model.plotAlpha(block=True,show=True)
# absorption_model.plotAlpha(block=True,show=True)
index_model = vam.medium.IndexModel(coord_vec, type = 'interpolation', form='homogeneous_r2r', n_sur=config.n_sur, n=config.n_mat, rho_grid=target_geo.transform.rho_g_w, r_mid=target_geo.transform.r_mid, tau=target_geo.transform.tau)
# index_model.plotIndex(block=True,show=True)
# index_model = vam.medium.IndexModel(coord_vec, type = 'analytical', form='homogeneous')
# response_model = vam.response.ResponseModel(B=10)
response_model = vam.response.ResponseModel(B=10)
# response_model = vam.response.ResponseModel(form='linear')
# x = np.linspace(0,1,100)
# y = response_model.map(x)
# fig, axs = plt.subplots(1,1)
# axs.plot(x,y)
# plt.show()
# %%
matrix_sino2uw_filename = os.path.join(args.matrix_dir, f'matrix_sino2uw_{config.matrixStr()}.pt')
angles = np.array([269.9])

if os.path.exists(matrix_sino2uw_filename):
    print('matrix_sino2uw is found')
    P_matrix_sino2uw = torch.load(matrix_sino2uw_filename,map_location=device)
    proj_geo_al = vam.geometry.ProjectionGeometry(angles,
                                            ray_type='algebraic_r2r',
                                            CUDA=True,
                                            loading_path_for_matrix = matrix_sino2uw_filename
                                            )
    
    P_al = vam.projector.pyTorchAlgebraicPropagationR2R.PyTorchAlgebraicR2RPropagator(target_geo, proj_geo_al,output_torch_tensor=False)

    # test_sino = torch.zeros((target_geo.nX,P_al.n_fold_blocks,target_geo.nZ),device=device)
    # test_sino[20:30,:,0] = 1
    # test_sino[220:230,:,0] = 2
    # test_sino[420:430,:,0] = 3
    # test_sino[620:630,:,0] = 4
    # test_sino[20:30,:,0] = 10
    # test_sino[220:230,:,0] = 20
    # test_sino[420:430,:,0] = 30
    # test_sino[620:630,:,0] = 40
    # test_sino = torch.ones((target_geo.nX,P_al.n_fold_blocks,target_geo.nZ),device=device)
    # test_recon = P_al.backward(test_sino)
    # fig, axs = plt.subplots(1,1)
    # axs.imshow(test_recon[:,:,0])
    # plt.show()
    # test_sino2 = P_al.forward(target_geo.array)
    # test_recon2 = P_al.backward(test_sino2)
    # test_recon = P_rt.backward(test_sino.ravel())
    # test_recon = test_recon.reshape(target_geo.transform.l_g_uw.shape).cpu().numpy()
    # test_recon = test_recon.reshape(target_geo.array.shape).cpu().numpy()
    # test_recon = test_recon.reshape(target_geo.array.shape).cpu().numpy()
    
    # fig, axs = plt.subplots(1,1)
    # axs.imshow(test_sino2[:,:,0])
    # # plt.show()
    # fig, axs = plt.subplots(1,1)
    # axs.imshow(test_recon2[:,:,0])
    # plt.show()
else:
    proj_geo_rt = vam.geometry.ProjectionGeometry(angles,
                                                ray_type='ray_trace',
                                                CUDA=True,
                                                index_model = index_model,
                                                attenuation_model = attenuation_model,
                                                absorption_model = absorption_model,
                                                ray_trace_method = 'eikonal',
                                                eikonal_parametrization = 'physical_path_length',
                                                ray_trace_ode_solver = 'forward_symplectic_euler',
                                                ray_trace_ray_config = 'r2r',
                                                inclination_angle = 0.0,
                                                ray_density = 1
                                                )

    #Direct call high level function
    P_rt = vam.projector.pyTorchRayTraceR2R.PyTorchRayTracingPropagator(target_geo, proj_geo_rt, output_torch_tensor=True)

    # test_sino = torch.zeros(P_rt.ray_state.sino_shape,device=device)
    # test_sino[20:30,0,0] = 1
    # test_sino = torch.ones(P_rt.ray_state.sino_shape,device=device) this is the line used
    # test_sino[0::40,0,:] = 1
    
    # test_recon = P_rt.backward(test_sino.ravel()) this is the line used
    # test_recon = test_recon.reshape(target_geo.transform.x_g_w.shape).cpu().numpy()
    # test_recon = test_recon.reshape(target_geo.transform.x_g_uw.shape).cpu().numpy() this is the line used
    # test_recon = test_recon.reshape(target_geo.array.shape).cpu().numpy()

    # fig, axs = plt.subplots(1,1)
    # axs.imshow(test_recon[:,:,0])
    # plt.show()

    P_matrix_sino2uw = P_rt.buildPropagationMatrix()
    torch.save(P_matrix_sino2uw,matrix_sino2uw_filename)
    sys.exit()
# %%
print('start optimization initial guess')
print("Matrix shape:", target_geo.array.shape)
g0 = P_al.forward(response_model.map_inv(target_geo.array))

# g0 = vam.util.data.filterSinogram(g0, 'ram-lak')
# g0 *= np.pi/(2*g0.shape[1]) #This scales the filtered backpropagation results correctly such that original target scale is maintained.
# g0 /= g0.shape[1]
# r0 = P_al.backward(g0)
g0 /= P_al.n_fold_blocks*100
# g0 /= np.sum(r0)
# g0 /= np.max(r0)
print(f"3D n_fold_blocks: {P_al.n_fold_blocks}")
print(f"3D g0 mean before scaling: {np.mean(g0) * P_al.n_fold_blocks * 100:.6f}")
print(f"3D g0 mean after scaling: {np.mean(g0):.8f}")
r0_3d = P_al.backward(g0)
in_part_3d = target_geo.array > 0
print(f"3D r0 mean in-part: {np.mean(r0_3d[in_part_3d]):.6f}")
print(f"3D r0 mean out-part: {np.mean(r0_3d[~in_part_3d]):.6f}")
print(f"3D r0 max: {np.max(r0_3d):.6f}")
#%%
data_2D = g0[:,g0.shape[1]//2,:]  # (detector, z) at the middle angle; no squeeze, so one z slice stays 2D
plt.imshow(data_2D.T, aspect='auto', cmap='viridis')  # Transpose, show image
plt.colorbar()
plt.show()
# %%
# np.save(rf'{test_string}_initializationsinogram.npy',g0)
fig, axs = plt.subplots(1,1)
im_sino = axs.imshow(g0[:,:,target_geo.array.shape[2]//2])
axs.set_xlabel(r'$\theta$ []')
axs.set_ylabel(r'$\rho$ [cm]')
vam.util.matplotlib.addColorbar(im_sino)
fig.savefig(os.path.join(directory,'initializationsinogram.png'),dpi=600)

# fig, axs = plt.subplots(1,1)
# im_g0 = axs.imshow(g0[:,:,0])
# axs.set_xlabel(r'$L$')
# axs.set_ylabel(r'$\rho$') 
# vam.util.matplotlib.addColorbar(im_g0)
# plt.show()

weight = np.ones_like(target_geo.array)
eps_map = np.ones_like(target_geo.array)

eps_map[target_geo.array > 0] = 0.01
eps_map[target_geo.array == 0] = 0.05

weight[target_geo.array>0] = 3
# weight[target_geo.array==0] = 3

optimizer_params = vam.optimize.Options(method='BCLP',
                                        n_iter = 250,
                                        filter='ram-lak',
                                        verbose='plot',
                                        p=3,
                                        learning_rate = 0.05,
                                        regularize_l = 0,
                                        # p=2,
                                        # learning_rate = 0.0005,
                                        # regularize_l = 0.05,
                                        momentum = 0.5,
                                        response_model = response_model,
                                        # eps=0.01,
                                        eps=eps_map,
                                        g0=g0,
                                        weight=weight,
                                        bit_depth=8,
                                        # save_img_path=rf'test_output\{config}'
                                        ) #Working

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()  # make sure all previous GPU ops are done
start = time.time()
opt_sino, opt_recon, opt_response, logs = vam.optimize.optimize(target_geo, proj_geo_al,optimizer_params, output = 'components')
plt.savefig(os.path.join(directory,'optimizationplot.png'),dpi=600)
end = time.time()
torch.cuda.synchronize()
peak_mem_bytes = torch.cuda.max_memory_allocated()
peak_mem_mb = peak_mem_bytes / 1024 ** 2

print("sino size:", opt_sino.shape)
print("sino unique values:", np.unique(opt_sino[:, opt_sino.shape[1]//2, :]).size)
print("sino data type:", opt_sino.dtype)
# %%
print(f"Peak GPU memory used: {peak_mem_mb:.2f} MB")
print(f"Elapsed time: {end - start:.2f} seconds")

np.save(os.path.join(directory,'sinogram.npy'),opt_sino)
np.save(os.path.join(directory,'reconstruction.npy'),opt_recon)
np.save(os.path.join(directory,'response.npy'),opt_response)

fig, axs = plt.subplots(1,1)
im_sino = axs.imshow(opt_sino[:,:,target_geo.array.shape[2]//2])
axs.set_xlabel(r'$\theta$ []')
axs.set_ylabel(r'$\rho$ []')
vam.util.matplotlib.addColorbar(im_sino)
fig.tight_layout()
fig.savefig(os.path.join(directory,'sinogram.png'),dpi=600)

# plt.show()
fig, axs = plt.subplots(1,1)
im_recon = axs.imshow(opt_recon[:,:,target_geo.array.shape[2]//2],extent=[0,target_geo.nL_total/target_geo.transform.spatial_sampling_rate_l,target_geo.transform.r_o,target_geo.transform.r_i])
# im_recon.set_clim(0,1)
axs.set_xlabel(r'$L$ [cm]')
axs.set_ylabel(r'$\rho$ [cm]')
vam.util.matplotlib.addColorbar(im_recon)
fig.tight_layout()
fig.savefig(os.path.join(directory,'reconstruction.png'),dpi=600)

fig, axs = plt.subplots(1,1)
im_recon = axs.imshow(opt_recon[:,0:200,target_geo.array.shape[2]//2],extent=[0,200/target_geo.transform.spatial_sampling_rate_l,target_geo.transform.r_o,target_geo.transform.r_i])
# im_recon.set_clim(0,1)
axs.set_xlabel(r'$L$ [cm]')
axs.set_ylabel(r'$\rho$ [cm]')
vam.util.matplotlib.addColorbar(im_recon)
fig.tight_layout()
fig.savefig(os.path.join(directory,'reconstruction_zoom.png'),dpi=600)
plt.show()
# %%