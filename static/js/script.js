let chatHistory = document.getElementById('chatHistory');
let userInput = document.getElementById('userInput');

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = `
        <div class="message-content">${text}</div>
        ${sender === 'bot' ? '<div class="status"><i class="fas fa-robot"></i></div>' : ''}
    `;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;

    addMessage(question, 'user');
    userInput.value = '';
    autoResize(userInput);

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        if (data.status === 'success') {
            addMessage(data.answer, 'bot');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        addMessage(`回答生成失败：${error.message}`, 'bot');
    }
}

// 文件上传处理
document.getElementById('fileInput').addEventListener('change', async function(e) {
    const files = e.target.files;
    if (files.length === 0) return;

    const formData = new FormData();
    for (let file of files) {
        formData.append('files', file);
    }

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (result.status === 'success') {
            addMessage(`知识库已更新，新增 ${result.message} 个文档块`, 'system');
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        alert(`上传失败：${error.message}`);
    }
});

// 回车发送
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});