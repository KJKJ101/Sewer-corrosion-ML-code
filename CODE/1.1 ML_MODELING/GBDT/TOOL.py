import os

# =========================
# 1. 文件夹路径
# =========================
folder_path = r"E:\Polimi_jiang\ML2024.01\pythonProject1\2026.03-可解释性下水道腐蚀\2025.1.13-可解释性实操（正确数据）\实操\r\1.1 ML_建模\GBDT\Model_GBDT"

# =========================
# 2. 输出文件
# =========================
output_file = os.path.join(folder_path, "r-Optuna_parameters_all_1-10.txt")

# =========================
# 3. 汇总
# =========================
with open(output_file, "w", encoding="utf-8") as outfile:

    for i in range(1, 11):

        file_name = f"Optuna_parameters_{i}.txt"
        file_path = os.path.join(folder_path, file_name)

        outfile.write("=" * 80 + "\n")
        outfile.write(f"Random split {i}\n")
        outfile.write(f"Source file: {file_name}\n")
        outfile.write("=" * 80 + "\n\n")

        if os.path.exists(file_path):

            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read()

            outfile.write(content)
            outfile.write("\n\n")

        else:
            outfile.write(f"WARNING: File not found: {file_name}\n\n")

print("汇总完成！")
print(f"保存位置：{output_file}")