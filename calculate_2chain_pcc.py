import pandas as pd
from scipy.stats import pearsonr
import os

def calculate_pearson_correlation(csv_file, pdb_list):
    # 读取CSV文件
    df = pd.read_csv(csv_file, header=None, names=['PDB', 'Predicted', 'Actual'])

    # 过滤出在给定列表中的PDB记录
    filtered_df = df[df['PDB'].isin(pdb_list)]

    # 检查是否存在记录
    if filtered_df.empty:
        print("没有匹配的PDB记录。")
        return None

    # 计算皮尔逊相关系数
    correlation, p_value = pearsonr(filtered_df['Predicted'], filtered_df['Actual'])

    return correlation

csv_file = '/public/mxp/xiejun/py_project/PPI_affinity/runs/run_11_final/attempt9_nompnn_enc_encpre4/evaluate_test.csv'
pdb_list = os.listdir("/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/2chain_test")  # 你的PDB名称列表
pdb_list=[a.replace(".ent.pdb","") for a in pdb_list]
print(len(pdb_list))
correlation = calculate_pearson_correlation(csv_file, pdb_list)

if correlation is not None:
    print(f"选定PDB记录的预测值和真实值之间的皮尔逊相关系数: {correlation}")