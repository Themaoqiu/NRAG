from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.llm import DeepseekChat  
from RAG.Embeddings import DeepSeekEmbedding

docs = ReadFiles(r'D:\software\VScode\python\Mathematical modeling\25_Mathorcup_C').get_content(max_token_len=500, cover_content=50)

# 初始化向量数据库
vector = VectorStore(docs)

embedding = DeepSeekEmbedding(is_api=False)

# 生成并存储向量
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path=r'D:\software\VScode\python\NRAG\RAG\database')

# 加载已有数据库
vector = VectorStore()
vector.load_vector(r'D:\software\VScode\python\NRAG\RAG\database')

# 初始化DeepSeek聊天模型
model = DeepseekChat(model="deepseek-chat")

question = '题干要求音频文件的什么？'

content = vector.query(
    question, 
    EmbeddingModel=embedding,
    k=3 
)[0]

# 生成回答
response = model.chat(
    prompt=question,
    history=[],
    content=content,
    temperature=0.3
)

print("问题：", question)
print("参考内容：", content)
print("模型回答：", response)