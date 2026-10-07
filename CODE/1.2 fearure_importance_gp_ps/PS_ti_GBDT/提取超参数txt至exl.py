import pandas as pd
import ast  # 用于将字符串转化为字典
import os

# 读取 txt 文件
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

# 解析 txt 中的超参数数据
def parse_hyperparameters(lines):
    hyperparameters_list = []
    for line in lines:
        # 找到包含超参数字典的行
        if line.strip().startswith("{"):
            # 使用 ast.literal_eval 将字符串转换为字典
            try:
                params = ast.literal_eval(line.strip())  # 转换为字典
                hyperparameters_list.append(params)
            except (ValueError, SyntaxError):
                pass  # 如果无法解析为字典，跳过该行
    return hyperparameters_list

# 将超参数数据保存到 Excel 文件
def save_to_excel(hyperparameters_list, output_file):
    # 转换为 DataFrame
    df = pd.DataFrame(hyperparameters_list)
    # 保存到 Excel 文件
    df.to_excel(output_file, index=False)

# 处理多个文件
def process_multiple_files(input_dir, output_file):
    all_hyperparameters = []  # 存储所有文件的超参数数据
    for i in range(1, 11):  # 处理从 1 到 10 的文件
        file_path = os.path.join(input_dir, f'Optuna_parameters_{i}.txt')  # 构造文件路径
        print(f'Reading file: {file_path}')
        lines = read_txt_file(file_path)
        hyperparameters_list = parse_hyperparameters(lines)
        all_hyperparameters.extend(hyperparameters_list)  # 合并数据
    # 将所有超参数保存到一个 Excel 文件
    save_to_excel(all_hyperparameters, output_file)
    print(f'All hyperparameters have been saved to {output_file}')

# 主函数
def main():
    # 设置 txt 文件所在目录
    input_dir = r'F:\ML2024.01\pythonProject1\2025.1.13-可解释性实操（正确数据\实操\r\1.2 ML_特征重要性排序（分位置-训练）\液相-速率-GBDT（速率最佳）\Model_GBDT'  # 替换为实际目录
    output_excel_path = r'F:\ML2024.01\pythonProject1\2025.1.13-可解释性实操（正确数据\实操\r\1.2 ML_特征重要性排序（分位置-训练）\液相-速率-GBDT（速率最佳）\GBDT_parameters.xlsx'  # 输出 Excel 文件路径

    # 处理所有 10 个文件并保存到 Excel
    process_multiple_files(input_dir, output_excel_path)

if __name__ == "__main__":
    main()
