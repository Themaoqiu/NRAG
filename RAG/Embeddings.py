from typing import List
import os
import numpy as np
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer  # 新增本地模型依赖

class BaseEmbeddings:
    """基类定义需保持与原始结构一致"""
    def __init__(self, path: str = '', is_api: bool = True):
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError("基类方法需实现")
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

class DeepSeekEmbedding(BaseEmbeddings):
    """
    DeepSeek Embedding服务封装（修正版）
    """
    def __init__(self, path: str = '', is_api: bool = True):
        super().__init__(path, is_api)  # 修复：必须调用父类初始化
        if self.is_api:
            self.client = OpenAI(
                api_key="sk-66de080296ae49018e1cf680f4b60aef",
                base_url="https://api.deepseek.com/v1"  # 修正API地址格式
            )
        else:
            # 加载本地开源模型
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
            self.model = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
    
    def get_embedding(self, text: str, model: str = "embedding-2") -> List[float]:
        if self.is_api:
            # API模式
            text = text.replace("\n", " ")  # 根据需求决定是否保留换行符
            response = self.client.embeddings.create(
                input=[text], 
                model=model
            )
            return response.data[0].embedding
        else:
            # 本地模型模式
            return self._local_embedding(text)
    
    def _local_embedding(self, text: str) -> List[float]:
        """使用BGE模型生成本地嵌入"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings[0].tolist()  # 转换为列表

# 使用示例
if __name__ == "__main__":
    # API模式
    # embedder = DeepSeekEmbedding(is_api=True)
    # api_vec = embedder.get_embedding("测试文本")
    
    # 本地模式
    local_embedder = DeepSeekEmbedding(is_api=False)
    local_vec = local_embedder.get_embedding("测试文本")
    print("测试结束")