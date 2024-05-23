## Mednist clasification

import nitrain as nt
from nitrain import samplers, readers, transforms as tx
import ants

# download data and unzip from this link:
# https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz
mednist_path = '~/Downloads/MedNIST/'

dataset = nt.Dataset(inputs=readers.ImageReader('*/*.jpeg'),
                     outputs=readers.FolderNameReader('*/*.jpeg', format='integer'),
                     base_dir=mednist_path)
