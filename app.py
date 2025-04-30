from flask import Flask, request, render_template, jsonify
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.llm import DeepseekChat
from RAG.Embeddings import DeepSeekEmbedding
import os

app = Flask(__name__)

# 初始化模型
EMBED_MODEL = DeepSeekEmbedding(is_api=False)
LLM_MODEL = DeepseekChat(model="deepseek-chat")
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), 'database')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

def initialize_database():
    """自动初始化数据库逻辑"""
    if not os.path.exists(os.path.join(DEFAULT_DB_PATH, 'docment.json')):
        # 处理默认文档（可选）
        # default_docs = ReadFiles("default_docs").get_content()
        # vector = VectorStore(default_docs)
        # vector.get_vector(EMBED_MODEL)
        # vector.persist(DEFAULT_DB_PATH)
        pass

@app.route('/')
def home():
    initialize_database()
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def handle_question():
    data = request.get_json()
    question = data['question']
    
    try:
        # 加载数据库
        vector = VectorStore()
        vector.load_vector(DEFAULT_DB_PATH)
        
        # 检索上下文
        context = vector.query(question, EMBED_MODEL, k=3)[0]
        
        # 生成回答
        response = LLM_MODEL.chat(
            prompt=question,
            history=[],
            content=context,
            temperature=0.3
        )
        
        return jsonify({
            "status": "success",
            "answer": response,
            "context": context
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"})
    
    try:
        # 保存文件
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # 处理文档
        docs = ReadFiles(UPLOAD_FOLDER).get_content(
            max_token_len=500, 
            cover_content=50
        )
        
        # 更新数据库
        vector = VectorStore(docs)
        vector.get_vector(EMBED_MODEL)
        vector.persist(DEFAULT_DB_PATH)
        
        # 清理上传文件（可选）
        # os.remove(file_path)
        
        return jsonify({
            "status": "success",
            "message": f"成功处理 {len(docs)} 个文档块"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)