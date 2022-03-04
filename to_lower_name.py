import shutil
import os

def tolower_name(dir_abs_path, file_):
    src_path = os.path.join(dir_abs_path, file_)
    dst_path = src_path.replace(file_, file_.lower()).replace(' ', '_')
    shutil.move(src_path, dst_path)

if __name__ == "__main__":
    dir_list = ['ssp', 'color_synthetic', 'layout_estimation']
    root_path = os.getcwd()

    for dir_ in dir_list:
        dir_abs_path = os.path.join(root_path, dir_)
        file_list = os.listdir(dir_)
        [tolower_name(dir_abs_path, file_) for file_ in file_list]