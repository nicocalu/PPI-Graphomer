import os
import subprocess
import numpy as np
from scipy.stats import pearsonr
import re
import pandas as pd


index_file = os.path.join("/public/mxp/xiejun/py_project/PP", "index", "INDEX_general_PP.2020")

# 获取亲和力信息
# PDBbind的获取方式，这里计算的是以10为底的log，在后面需要换底公式换掉
affinity_dict1 = {}
with open(index_file, 'r') as f:
    lines = f.readlines()
    for line in lines[6:]:
        tokens = line.split()
        protein_name = tokens[0].strip()
        # if protein_name=="5nvl":
        #     print("here")
        # 匹配 Kd, Ki, 或 IC50
        match = re.search(r'(Kd|Ki|IC50)([=<>~])([\d\.]+)([munpfM]+)', line)
        if match:
            measure_type = match.group(1)
            operator = match.group(2)
            value = float(match.group(3))
            unit = match.group(4)
            # 单位转换成 M（摩尔）
            unit_multiplier = {
                'mM': 1e-3,
                'uM': 1e-6,
                'nM': 1e-9,
                'pM': 1e-12,
                'fM': 1e-15
            }
            value_in_molar = value * unit_multiplier.get(unit, 1)  # 默认为 Mol

            # 计算以10为底的log值（pKa）
            if operator == '=' or operator == '~' or operator == '>':
                pKa_value = -np.log(value_in_molar)*0.592
            elif operator == '<':
                # 如果是 '<'，则取 "<" 值更保守的一种处理方式
                pKa_value = -np.log(value_in_molar)*0.592

            affinity_dict1[protein_name] = pKa_value

symbol="bm79"

foldx_dict={}
with open("/public/mxp/xiejun/py_project/PPI_affinity/rosetta_result/Interaction_output_AC_"+symbol+".fxout","r") as rf:
    lines=rf.readlines()
    for i,line in enumerate(lines):
        if i<9:
            continue
        line_list=line.split("\t")
        foldx_dict[line_list[0].split("/")[-1]]=float(line_list[3])


# benchmark79的亲和力数据
# 读取CSV文件
csv_file = '/public/mxp/xiejun/py_project/PPI_affinity/elife-07454-supp4-v4.csv'
data = pd.read_csv(csv_file)
# 创建包含 PDB 名称和亲和力数值的字典
affinity_dict3 = dict(zip(data.iloc[:, 0].apply(lambda x: x.replace(".pdb", "")), data.iloc[:, 1]))

affinity_dict={}
affinity_dict.update(affinity_dict1)
affinity_dict.update(affinity_dict3)


# 指定Rosetta的路径和PDB文件夹
# pdb_folder = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/benchmark79/"
# result_sc_path = "/public/mxp/xiejun/py_project/PPI_affinity/rosetta_result/benchmark79_sc/"
pdb_folder = "/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/2chain_"+symbol+"/"
result_sc_path = "/public/mxp/xiejun/py_project/PPI_affinity/rosetta_result/"+symbol+"_Interface_sc2/"


energy_out_list=[]
true_affinity=[]
for protein in os.listdir(pdb_folder):
    outfile=result_sc_path+protein+"/"+"interface_score.sc"
    # outfile=result_sc_path+protein.split(".pd")[0]+".sc"
    with open(outfile,"r") as rf:
        lines=rf.readlines()
        for index,line in enumerate(lines):
            # 跳过前1行
            if index<2:
                continue
            energy_out_list.append(float(re.split(r"[ ]+", line)[21]))
    # foldx分数
    # energy_out_list.append(foldx_dict[protein])
    # 真实值
    true_affinity.append(-affinity_dict[protein.split(".")[0]])


# [78.3392, 21.1608, 24.3767, 58.1703, 43.1543, 20.6342, 19.6766, 10.3637, 67.9962, 36.2332, 6.11165, 107.102, 26.5714, 5.13, 80.8053, 303.619, 25.4963, 10.34, 13.2873, 16.168, 33.0786, 21.4284, 23.67, 0.63, 170.598, 33.5819, 14.7482, 61.3045, 17.1436, 11.46, 8.17842, 52.91, 19.0313, 5.72615, 1.68178, 20.2609, 88.4946, 107.379, 96.3494, 18.1995, 29.1308, 22.1099, 65.3511, 67.6013, 9.07848, 27.3215, 55.868, 27.64, 30.2272, 6.6699, 28.3489, 39.04, 68.41, 43.4393, 11.794, 44.883, 23.0403, 41.563, 43.1857, ...]
# [-9.97, -10.200, -10.758, -11.2074, -12.5705, -10.626, -9.301, -6.575,   -6.841, -15.6689, -6.9781, -14.018, -10.362, -7.830, -10.864989080523301, -10.494699869528311, -9.083274142176323, -7.768439119423362, -14.795242561497465, -9.874687024914222, -18.02310364495164, -7.893185969962229, -9.31787825542094, -6.702804622374054, -12.810617488741775, -8.122358623870689, -8.532701754762176, -10.948004850578023, -8.215412489315957, -5.344587138587879, -9.728221386312425, -7.8646503256860445, -10.494699869528311, -5.994965613479399, -14.584090994685736, -11.11619456723153, -8.389933817126579, -7.490196970909887, -12.268173375472275, -9.681460246811797, -10.70585143634004, -7.343477961878374, -4.089391125157425, -8.407094438955545, -10.438276243084152, -7.548473494015332, -8.122358623870689, -4.228938746601898, -8.891534150475804, -8.999468512097826, -9.496351689014736, -10.494699869528311, -9.572278255644754, -12.211749749028113, -11.577740980684256, -9.915530804834496, -8.696586332167385, -7.358095988531875, -8.589125381206337, ...]



print(len(true_affinity))
# 计算皮尔逊相关系数
if len(energy_out_list) > 1:  # 至少需要两个数据点
    correlation, _ = pearsonr(true_affinity, energy_out_list)
    print(f"皮尔逊相关系数: {correlation}")
else:
    print("数据点不足，无法计算皮尔逊相关系数")