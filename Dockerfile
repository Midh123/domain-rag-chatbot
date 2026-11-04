FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
# Pass the key at run time:  docker run -p 8501:8501 -e GROQ_API_KEY=your_key rag-chatbot
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
