import os
import re
import json
from typing import List, Dict
from tqdm import tqdm
import tiktoken
import fitz
from pypdf import PdfReader, errors

class ReadFiles:
    """
    文件读取与分块处理类（DeepSeek适配版）
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()
        self.enc = tiktoken.get_encoding("cl100k_base")

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 2000, cover_content: int = 200) -> List[str]:
        """获取分块内容（适配长上下文模型）"""
        docs = []
        for file in tqdm(self.file_list, desc="Processing files"):
            try:
                content = self.read_file_content(file)
                chunks = self.get_chunk(content, max_token_len, cover_content)
                docs.extend(chunks)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        return docs

    def get_chunk(self, text: str, max_token_len: int, cover_content: int) -> List[str]:
        """改进的分块算法（保留语义完整性）"""
        # 预处理
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)  # 按句子分割
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            sent_len = len(self.enc.encode(sent))
            
            # 长句子单独成块
            if sent_len > max_token_len:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.extend(self._split_long_sentence(sent, max_token_len))
                continue
                
            # 正常分块
            if current_len + sent_len <= max_token_len:
                current_chunk.append(sent)
                current_len += sent_len
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-int(len(current_chunk)*0.2):]  # 20%重叠
                current_chunk.append(sent)
                current_len = sum(len(self.enc.encode(s)) for s in current_chunk)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _split_long_sentence(self, sentence: str, max_len: int) -> List[str]:
        """处理超长句子"""
        words = sentence.split()
        chunks = []
        current_chunk = []
        current_len = 0
        
        for word in words:
            word_len = len(self.enc.encode(word))
            if word_len > max_len:
                continue  # 跳过异常长单词
                
            if current_len + word_len <= max_len:
                current_chunk.append(word)
                current_len += word_len
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_len = word_len
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def read_file_content(self, file_path: str) -> str:
        """改进的文件读取方法"""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                return self.read_pdf(file_path)
            elif ext == '.md':
                return self.read_markdown(file_path)
            elif ext == '.txt':
                return self.read_text(file_path)
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {str(e)}")

    def read_pdf(self, file_path: str) -> str:
        """改进的PDF解析方法"""
        text = ""
        try:
            # 优先使用PyMuPDF提取文本
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text("text")
        except Exception:
            # 回退到pypdf
            try:
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    text = ''.join([page.extract_text() or '' for page in reader.pages])
            except errors.PdfReadError as e:
                raise RuntimeError(f"PDF解析失败: {str(e)}")
        
        # 清理PDF特殊字符
        return re.sub(r'\x0c', '', text)  # 移除换页符

class Documents:
    """数据处理适配器"""
    def __init__(self, path: str = ''):
        self.path = path
    
    def get_content(self) -> List[Dict]:
        """生成DeepSeek兼容格式"""
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [{
                "id": str(item.get("id", "")),
                "text": item.get("content", ""),
                "metadata": item.get("meta", {})
            } for item in data]