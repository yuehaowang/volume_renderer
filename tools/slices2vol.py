'''
Generate binary volume data from slices (images)
Data are downloaded from http://graphics.stanford.edu/data/voldata/
'''


import os
import sys
from PIL import Image
import array

# Fixed the width of bounding box
BBOX_SIZE_X = 5.0

def main():
    in_data_path = sys.argv[1]
    out_path = sys.argv[2]
    vol_size = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

    slices_files = sorted(os.listdir(in_data_path))

    im = Image.open(os.path.join(in_data_path, slices_files[0]))
    w, h = im.size

    grid_dim = (w, h, len(slices_files))
    bbox_size = [
        BBOX_SIZE_X, (BBOX_SIZE_X / grid_dim[0]) * (vol_size[1] / vol_size[0]) * grid_dim[1],
        (BBOX_SIZE_X / grid_dim[0]) * (vol_size[2] / vol_size[0]) * grid_dim[2]
    ]

    print('Volume dimension:', grid_dim)
    print('Volume BBox:', bbox_size)

    # Write grid dimension (integers)
    out_f = open(out_path, 'wb')
    i_out_values = array.array('i')
    i_data_list = [grid_dim[0], grid_dim[1], grid_dim[2]]
    i_out_values.fromlist(i_data_list)
    i_out_values.tofile(out_f)
    out_f.close()


    # Write bbox size and volume data (floats)
    out_f = open(out_path, 'ab')
    f_out_values = array.array('f')
    f_data_list = [bbox_size[0], bbox_size[1], bbox_size[2]]
    for fn in slices_files:
        f_path = os.path.join(in_data_path, fn)
        im = Image.open(f_path)

        w, h = im.size

        for i in range(w):
            for j in range(h):
                f_data_list.append(im.getpixel((i, j)) / 255.0)

    f_out_values.fromlist(f_data_list)
    f_out_values.tofile(out_f)
    out_f.close()



if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: python slices2vol.py /path/to/data /path/to/output voxel_size_x_ratio voxel_size_y_ratio voxel_size_z_ratio')
        print('e.g., python slices2vol.py data/cthead-8bit/ resources/cthead.bin 1 1 2')
        exit(-1)
    
    main()
