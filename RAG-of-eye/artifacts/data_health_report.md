# 🏥 RAG-of-Eye 数据健康体检报告
**生成时间**: 2026-01-12 20:15:37

## 1. 📊 数据覆盖率全景图 (Data Coverage Matrix)
> 图例: ❌ = 缺失 | ⚠️ = 样本不足(<50) | ✅ = 充足

| Group | Age 6 | Age 7 | Age 8 | Age 9 | Age 10 | Age 11 | Age 12 | Age 13 | Age 14 | Age 15 | Age 16 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Asian_Female_Atropine_Low** | ✅ (240) | ❌ | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Asian_Female_Ortho_K** | ✅ (120) | ✅ (120) | ❌ | ✅ (120) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Asian_Female_Natural** | ❌ | ❌ | ✅ (120) | ❌ | ✅ (120) | ✅ (120) | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Asian_Male_Atropine_Low** | ✅ (120) | ❌ | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ❌ | ❌ | ✅ (120) | ❌ | ✅ (120) |
| **Asian_Male_Ortho_K** | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ❌ | ✅ (120) | ✅ (120) | ❌ | ❌ |
| **Asian_Male_Natural** | ❌ | ❌ | ❌ | ✅ (120) | ✅ (120) | ❌ | ❌ | ✅ (120) | ❌ | ❌ | ✅ (120) |
| **Caucasian_Female_Atropine_Low** | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ❌ | ❌ | ❌ |
| **Caucasian_Female_Ortho_K** | ✅ (120) | ❌ | ❌ | ✅ (120) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (120) | ❌ |
| **Caucasian_Female_Natural** | ❌ | ❌ | ✅ (438) | ✅ (120) | ✅ (120) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Caucasian_Male_Natural** | ✅ (120) | ❌ | ✅ (120) | ❌ | ❌ | ✅ (120) | ✅ (120) | ❌ | ❌ | ❌ | ❌ |
| **Caucasian_Male_Atropine_Low** | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) | ✅ (120) |
| **Caucasian_Male_Ortho_K** | ✅ (120) | ✅ (120) | ✅ (120) | ❌ | ❌ | ✅ (120) | ❌ | ✅ (120) | ❌ | ✅ (120) | ✅ (120) |
| **Caucasian_Male_Defocus_Glasses** | ❌ | ❌ | ✅ (120) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

## 2. 🚨 严重阻断性问题 (Critical Issues)
✅ **太棒了！未发现严重的阻断性数据问题。**

## 4. 📝 自动生成的下一步指令 (Action Plan)
**请复制以下指令到终端运行，以填补数据空缺：**

🔴 **严重缺失**: Asian_Female_Atropine_Low 缺失大量数据。
   > 建议指令: `python system/mining_task.py --ethnicity Asian --gender Female --treatment Atropine_Low --target_ages 7-16`
🔴 **严重缺失**: Asian_Female_Ortho_K 缺失大量数据。
   > 建议指令: `python system/mining_task.py --ethnicity Asian --gender Female --treatment Ortho_K --target_ages 8-16`
🔴 **严重缺失**: Asian_Female_Natural 缺失大量数据。
   > 建议指令: `python system/mining_task.py --ethnicity Asian --gender Female --treatment Natural --target_ages 6-16`
🟠 **填补空白**: Asian_Male_Atropine_Low 缺失年龄 7,12,13,15。
   > 建议指令: `python system/mining_task.py --ethnicity Asian --gender Male --treatment Atropine_Low --target_ages 7,12,13,15`
🟠 **填补空白**: Asian_Male_Ortho_K 缺失年龄 12,15,16。
   > 建议指令: `python system/mining_task.py --ethnicity Asian --gender Male --treatment Ortho_K --target_ages 12,15,16`
🔴 **严重缺失**: Asian_Male_Natural 缺失大量数据。
   > 建议指令: `python system/mining_task.py --ethnicity Asian --gender Male --treatment Natural --target_ages 6-15`
🟠 **填补空白**: Caucasian_Female_Atropine_Low 缺失年龄 14,15,16。
   > 建议指令: `python system/mining_task.py --ethnicity Caucasian --gender Female --treatment Atropine_Low --target_ages 14,15,16`
🔴 **严重缺失**: Caucasian_Female_Ortho_K 缺失大量数据。
   > 建议指令: `python system/mining_task.py --ethnicity Caucasian --gender Female --treatment Ortho_K --target_ages 7-16`
🔴 **严重缺失**: Caucasian_Female_Natural 缺失大量数据。
   > 建议指令: `python system/mining_task.py --ethnicity Caucasian --gender Female --treatment Natural --target_ages 6-16`
🔴 **严重缺失**: Caucasian_Male_Natural 缺失大量数据。
   > 建议指令: `python system/mining_task.py --ethnicity Caucasian --gender Male --treatment Natural --target_ages 7-16`
🟠 **填补空白**: Caucasian_Male_Ortho_K 缺失年龄 9,10,12,14。
   > 建议指令: `python system/mining_task.py --ethnicity Caucasian --gender Male --treatment Ortho_K --target_ages 9,10,12,14`
🔴 **严重缺失**: Caucasian_Male_Defocus_Glasses 缺失大量数据。
   > 建议指令: `python system/mining_task.py --ethnicity Caucasian --gender Male --treatment Defocus_Glasses --target_ages 6-16`
