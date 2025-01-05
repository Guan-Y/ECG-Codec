import multiprocessing as mp
import os.path as osp
import subprocess
from glob import glob

from tqdm import tqdm
import os

input_dir = "../datasets/mit-bih/*.atr"

ecg_data = sorted([osp.splitext(i)[0] for i in glob(input_dir)])
pbar = tqdm(total=len(ecg_data))

script_name = "datagen_mitbih.py"

# select one to uncomment to generate full data or segmented data
# generate_full_data = True
generate_full_data = False

gentest = eval("True")

if generate_full_data:
    '''use below for generating full data'''
    output_path = glob("../data/full_data/mit-bih/*/*/")
else:
    ''' 
        use this for generating segmented data
        segment length is defined in datagen_xxx
    '''
    output_path = glob("../data/train_data/*/*/*")

def clear_folder(folder_path):
    # 判断文件夹是否存在
    if not osp.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在")
        return

    # 判断文件夹是否为空
    if not os.listdir(folder_path):
        print(f"文件夹 '{folder_path}' 是空的")
        return

    # 删除文件夹下的所有文件
    for filename in os.listdir(folder_path):
        file_path = osp.join(folder_path, filename)
        try:
            if osp.isfile(file_path):
                os.remove(file_path)
                print(f"{file_path} 已删除")
            else:
                print(f"{file_path} 不是文件")
        except Exception as e:
            print(f"删除 {file_path} 时出错：{e}")


def run(file):
    params = ["python", script_name, "--file", file, "--full", str(generate_full_data)]
    subprocess.check_call(params)
    pbar.update(1)


if __name__ == "__main__":
    p = mp.Pool(processes=12)
    p.map(clear_folder, output_path)
    p.map(run, ecg_data)
