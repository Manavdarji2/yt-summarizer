# 🎥 yt-summarizer

A lightweight **YouTube Video Summarizer** that leverages **YouTube transcripts** and supporting documents to generate clear, structured summaries. Powered by **LLMs (Large Language Models)**, it enables both automated summarization and **query-based summaries** so you can extract exactly what you need without watching the full video.  

---

## ✨ Features

- 📄 **Transcript Extraction** – Automatically fetches YouTube transcripts (if available).  
- 📝 **Structured Summaries** – Organizes key points into concise sections.  
- 🔍 **Query-Based Summarization** – Ask questions about the video to get custom answers.  
- 📚 **Supporting Documents Integration** – Incorporate external resources to enrich summaries.  
- ⚡ **Lightweight & Flexible** – Minimal setup, works with any LLM backend.  

---

## 🚀 Getting Started

### 1. Clone the repo
```

git clone https://github.com/Manavdarji2/yt-summarizer.git
cd yt-summarizer

```

### 2. Install dependencies
```

pip install -r requirements.txt

```

### 3. Configure environment
Create a `.env` file to store your **API keys** (e.g., OpenAI, Anthropic, etc.):
```

GOOGLE_API_KEY=your_api_key_here

```

### 4. Run
```python

strealit run main.py

```


## 📂 Project Structure
```

yt-summarizer/
├── main.py                \# Main entry point
├──  langchain_youtube_video_loader.py 
                             \# Fetch YouTube transcripts
├── requirements.txt      \# Dependencies
├── README.md             \# Documentation
└── .env.example          \# Env template

```
