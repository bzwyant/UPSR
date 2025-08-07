import pathlib
from basicsr.utils import FileClient, imfrombytes, imwrite
from basicsr.utils.matlab_functions import imresize
from basicsr.data.data_util import scandir

script_dir = pathlib.Path(__file__)
root = script_dir.parent.parent.parent  # Go up to the root of the project

file_client = FileClient('disk')
gt_folder = script_dir / 'basicsr/datasets/galaxy_mnist/gt'
paths = sorted(list(scandir(gt_folder, full_path=True)))

for path in paths:
    img_bytes = file_client.get(path, 'gt')
    img_gt = imfrombytes(img_bytes, float32=True)
    img_lq = imresize(img_gt, 0.25)
    lq_path = path.replace('gt', 'lq_bicubic_matlab')
    # print(img_lq*255)
    imwrite(img_lq*255, lq_path)
    print(lq_path)
