# 📝 AI Blog Writing Agent

## 🚀 Project Title

**AI Blog Writing Agent using LangGraph**

An AI-powered multi-agent system that automatically researches, plans, and generates high-quality blog posts with citations and relevant images through an interactive Streamlit interface.

---

## ❓ Problem Statement

Writing informative and well-structured blogs requires extensive research, content planning, drafting, and collecting supporting references. This process is time-consuming and often requires switching between multiple tools.

The AI Blog Writing Agent automates the entire blogging workflow by using AI agents to plan the blog, perform web research, gather evidence, generate content, suggest relevant images, and present the final blog in a user-friendly interface.

---

## 🏗️ Architecture

The project is built using a multi-agent workflow powered by **LangGraph** and **LangChain**.

### Technologies Used

- **Python**
- **LangGraph**
- **LangChain**
- **OpenAI GPT**
- **Google Gemini**
- **Tavily Search API**
- **Streamlit**
- **Jupyter Notebook**
- **Markdown**

### Workflow

```
User
   │
   ▼
Streamlit Frontend
   │
   ▼
LangGraph Backend
   │
   ▼
Planner Agent
   │
   ├──► Research Agent (Tavily Search)
   ├──► Writer Agent (OpenAI / Gemini)
   └──► Image Agent
            │
            ▼
      Reducer Agent
            │
            ▼
Final Blog with Citations & Images
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/BLOGGING-AGENT.git
cd BLOGGING-AGENT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Run the application

```bash
streamlit run bwa_frontend.py
```

### 5. Open the application

Open the local URL displayed in your terminal (usually `http://localhost:8501`) and enter a blog topic to generate an AI-powered blog.

---

## 👩‍💻 Author

**Shivani Yadav**
