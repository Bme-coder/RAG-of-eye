import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import sys
import os
from dotenv import load_dotenv # 导入 dotenv

# ==========================================
# 核心修复区：强制加载上一级目录的 .env 文件
# ==========================================
# 1. 获取当前脚本所在目录 (system/test)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取上一级目录 (system)
parent_dir = os.path.dirname(current_dir)
# 3. 拼接 .env 路径
env_path = os.path.join(parent_dir, '.env')

# 4. 强制加载环境变量
print(f"正在加载环境变量: {env_path}")
load_dotenv(env_path)

# 5. 把上一级目录加入系统路径，确保能 import rag_core
sys.path.append(parent_dir)
# ==========================================

# 现在的 import 就能正常工作了
from rag_core import setup_global_settings
from llama_index.core import Settings

def init_environment():
    """初始化环境"""
    print("正在初始化 Embedding 模型...")
    # 这一步会读取刚才 load_dotenv 加载进来的 Key
    setup_global_settings() 
    return Settings.embed_model

def experiment_1_heatmap(embed_model):
    """
    论文实验一：相关性矩阵热力图
    """
    print("\n--- 正在运行实验 1: 生成相关性热力图 ---")
    
    queries = [
        "What is the progression rate of myopia?", 
        "Does outdoor activity reduce myopia risk?",
        "Side effects of high-dose atropine?" 
    ]
    
    documents = [
        "A1: Myopia progression is faster in younger children, around -1.0D/year.", 
        "A2: Time outdoors is a protective factor against myopia onset.",           
        "A3: High-concentration atropine causes photophobia and blur.",             
        "A4: Glaucoma causes irreversible damage to the optic nerve.",              
        "A5: Diabetes patients should monitor blood sugar levels."                  
    ]
    
    # 获取向量
    q_vecs = [embed_model.get_text_embedding(q) for q in queries]
    d_vecs = [embed_model.get_text_embedding(d) for d in documents]
    
    Q = np.array(q_vecs) 
    D = np.array(d_vecs) 
    
    # 矩阵乘法 (内积)
    similarity_matrix = np.dot(Q, D.T)
    
    # 画图
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.2)
    ax = sns.heatmap(
        similarity_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu",
        xticklabels=["Doc1(Rate)", "Doc2(Outdoor)", "Doc3(Atropine)", "Doc4(Glaucoma)", "Doc5(Diabetes)"],
        yticklabels=["Q1(Rate)", "Q2(Outdoor)", "Q3(Side Effect)"]
    )
    plt.title("Matrix Multiplication: Query-Document Correlation Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("paper_fig1_heatmap.png", dpi=300)
    print("✅ 图 1 已保存: paper_fig1_heatmap.png")

def experiment_2_pca_scatter(embed_model):
    """
    论文实验二：语义空间降维可视化
    """
    print("\n--- 正在运行实验 2: 生成 PCA 散点图 ---")
    
    corpus = {
        "Myopia (Medical)": [
            "Axial length elongation causes myopia.",
            "Orthokeratology lenses reshape the cornea.",
            "Atropine eye drops slow down progression.",
            "High myopia increases retinal detachment risk.",
            "Refractive error measurement in diopters.",
            "Concave lenses are used for correction.",
            "Scleral remodeling in nearsightedness."
        ],
        "Coding (Computer)": [
            "Python function definition using def.",
            "Recursive algorithm complexity is O(n).",
            "Matrix multiplication in NumPy.",
            "Deep learning requires GPU acceleration.",
            "Git commit and push to remote repository.",
            "Debugging runtime errors in code.",
            "SQL database query optimization."
        ]
    }
    
    all_texts = []
    labels = []
    
    for label, texts in corpus.items():
        for t in texts:
            all_texts.append(embed_model.get_text_embedding(t))
            labels.append(label)
            
    X = np.array(all_texts)
    
    # PCA 降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Category': labels
    })
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x='PC1', y='PC2', hue='Category', 
        style='Category', s=200, palette="deep"
    )
    plt.title("2D Projection of Text Embeddings (PCA Visualization)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("paper_fig2_pca.png", dpi=300)
    print("✅ 图 2 已保存: paper_fig2_pca.png")

if __name__ == "__main__":
    try:
        # 1. 初始化
        embed_model = init_environment()
        
        # 2. 运行实验
        experiment_1_heatmap(embed_model)
        experiment_2_pca_scatter(embed_model)
        
        print("\n🎉 论文插图生成完毕！请查看 test 文件夹下的 png 文件。")
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("建议检查: .env 文件是否存在？API Key 是否正确？")