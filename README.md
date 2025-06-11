# DocuQuery AI 📚

**Your Intelligent Document & Web Page Query Assistant**

---

## 🚀 Overview

DocuQuery AI is an advanced AI-powered application designed to intelligently process and answer questions from various document types (PDFs, Word documents, text files) and dynamic web pages (blogs, academic faculty directories). Leveraging the power of Large Language Models (LLMs) and advanced information retrieval techniques, DocuQuery AI acts as a smart research assistant, providing concise, structured, and context-specific answers directly from your provided sources.

## ✨ Features

* **Multi-Source Input:** Seamlessly ingest content from:
    * URLs (Blogs, News Articles, Academic Faculty Pages)
    * PDF Documents
    * Microsoft Word Documents (.docx)
    * Plain Text Files (.txt)
    * Direct Text Input
* **Intelligent Web Scraping:** Employs `Selenium` and `BeautifulSoup` for robust extraction from complex, dynamic websites, including specialized handling for academic faculty pages to identify names and roles.
* **Semantic Search (RAG):** Utilizes **Retrieval Augmented Generation (RAG)** by creating a highly efficient vector store (`FAISS`) from your documents, enabling the LLM to search for and retrieve only the most relevant information before generating an answer.
* **Context-Aware AI Assistant:** Powered by **Google Gemini 2.0 Flash**, configured with a custom prompt to act as an expert in IoT and Backend Development, providing structured answers tailored to the query type:
    * Concise summaries for general queries.
    * Bullet points with technical details and code snippets for technical questions.
    * Precise factual extraction (e.g., names and roles from faculty lists).
    * Comprehensive lists for enumeration queries.
* **Interactive Q&A Flow:**
    * **"Continue Answer":** Extend truncated answers or long lists with additional details.
    * **"Dive Deeper":** Obtain advanced, intricate technical insights related to the previous answer.
* **Rate Limiting:** Implements a token bucket mechanism to manage API calls to the LLM, ensuring compliance with usage limits and robust performance.
* **User-Friendly Interface:** Built with Streamlit for an intuitive and interactive web application experience.
* **Debug Mode:** Allows users to view the raw extracted text for verification and troubleshooting.

## 💡 Why DocuQuery AI over General LLMs (e.g., ChatGPT, Gemini Public)?

While general AI tools like ChatGPT or Gemini are powerful for broad knowledge, DocuQuery AI offers critical advantages for specific, controlled information retrieval:

1.  **Guaranteed Contextual Accuracy & No Hallucination:**
    * General LLMs draw from their vast training data, which can sometimes lead to 'hallucinations' or generic responses.
    * **DocuQuery AI answers *only* from the content you provide.** This ensures factual accuracy, prevents misinformation, and is vital for sensitive, proprietary, or highly specialized information where external knowledge is irrelevant or detrimental. Our specialized faculty list extraction is a prime example of this precise contextual awareness.

2.  **Deep Customization & Control:**
    * **Tailored Data Ingestion:** General LLMs cannot directly read specific PDF, DOCX files, or perform advanced web scraping on dynamic sites. DocuQuery AI handles these diverse input formats robustly.
    * **Fine-tuned Output:** We've configured the LLM's persona and prompt structure to deliver answers in a highly specific, useful format (summaries, bullet points, code, precise facts). This level of control over the AI's output is not available in generic chat interfaces.
    * **Specialized User Experience:** Features like "Continue Answer" and "Dive Deeper" are custom-built to provide a more guided and effective exploration of detailed information.

3.  **Efficiency & Potential for Privacy:**
    * Sending entire documents to public LLMs for every query can be expensive (token usage). DocuQuery AI first retrieves *only the most relevant snippets* from your data using its vector store, optimizing token usage and cost.
    * For sensitive or internal documents, sending content to public AI services may raise privacy concerns. DocuQuery AI's architecture provides a framework where data processing can be more controlled, potentially keeping more of your information on-premises or sending only minimal, anonymized snippets to the LLM.

In essence, DocuQuery AI is not just a general AI; it's your dedicated, specialized research assistant, expertly tuned to extract and present precise, reliable, and context-specific answers from *your* chosen documents and web sources.

## 🏗️ Architecture / How It Works

DocuQuery AI operates through a sophisticated pipeline:

1.  **Input & Extraction:** Users provide URLs, upload files (PDF, DOCX, TXT), or paste text. The `DocumentProcessor` intelligently extracts raw content, using specialized tools like `Selenium` for dynamic web pages and `PyPDF2`/`python-docx` for documents.
2.  **Text Chunking:** The extracted text is broken down into smaller, manageable "chunks" using `RecursiveCharacterTextSplitter`. This prevents information overload for the LLM and improves search relevance.
3.  **Embeddings Generation:** Each text chunk is converted into a high-dimensional numerical vector (an "embedding") by a `HuggingFaceEmbeddings` model. These embeddings capture the semantic meaning of the text.
4.  **Vector Store Creation:** All embeddings are stored in a super-fast, searchable database called a `FAISS` vector store. This "smart library" allows for rapid similarity searches.
5.  **Query & Retrieval:** When a user asks a question, the question itself is converted into an embedding. The FAISS vector store then efficiently identifies and retrieves the most relevant text chunks (based on embedding similarity) from the stored documents.
6.  **Answer Generation (RAG):** The retrieved relevant text chunks, along with the user's original question and a carefully crafted `PromptTemplate`, are fed to the `ChatGoogleGenerativeAI` (Gemini 2.0 Flash) model. The LLM then generates a concise, context-aware, and structured answer, adhering to the persona and output format rules defined in the prompt.
7.  **Interactive Experience:** Streamlit provides the web interface, managing session state to enable "Continue Answer" and "Dive Deeper" functionalities, allowing users to explore information incrementally.
8.  **Rate Limiting:** A `TokenBucket` mechanism ensures responsible API usage, preventing excessive calls to the Gemini LLM.

*For a visual representation of the architecture, please refer to the project's documentation or a separate diagram if available.*

## 🛠️ Prerequisites

Before you run DocuQuery AI, ensure you have the following installed on your system:

* **Python 3.9+** (recommended)
* **pip** (Python package installer)
* **Git** (for cloning the repository)
* **Google Gemini API Key:** You'll need an API key from Google AI Studio. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 🚀 Installation & Setup

Follow these steps to get DocuQuery AI up and running on your local machine:

### 1. Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone [https://github.com/YourGitHubUsername/DocuQuery-AI.git](https://github.com/YourGitHubUsername/DocuQuery-AI.git)
cd DocuQuery-AI
```
### 2. Create a Python Virtual Environment (Recommended)

It's good practice to use a virtual environment to manage project dependencies:

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**On Windows:**
```bash
.venv\Scripts\activate
```
**On MacOS/Linux:**
```bash
source .venv/bin/activate
```
### 4. Install dependencies:
```bash
pip install -r requirements.txt
```
### 5. Create `secret_api_keys.py` in the project root:
```python
 gemini_api_key = "YOUR_GEMINI_API_KEY"
```

## 📂 Project Structure
```
DocuQuery-AI/
├── .venv/                     # Python virtual environment (ignored by Git)
├── app.py                     # Main Streamlit application code
├── requirements.txt           # List of Python dependencies
├── secret_api_keys.py         # Stores your Gemini API key (ignored by Git)
├── .gitignore                 # Specifies files/folders to ignore for Git
├── extraction_debug.log       # Log file for extraction process (ignored by Git)
└── README.md                  # This README file
```
## 🏃  Usage
Start the application:
```bash
streamlit run app.py
```

🧑‍💻 Interact with the App

* **Select Input Type**: Link, PDF, Text, DOCX, TXT.
* **Provide Content**: Upload a file, paste text, or enter a URL.
* **Click "Proceed"**: This builds the vector store from your input.
* **Ask a Question**: Use the text input to ask about the content.
* **Click "Get Answer"**: Receive an LLM-generated, context-aware reply.
* **Use "Continue Answer"**: To reveal more if the answer was long.
* **Use "Dive Deeper"**: To extract more intricate insights.
* **Toggle Debug Mode**: In the sidebar to see raw extracted text.

## Dependencies
- `streamlit`: Web interface
- `langchain`: Document processing and QA chains
- `faiss-cpu`: Vector similarity search
- `PyPDF2`: PDF processing
- `python-docx`: DOCX processing
- `huggingface-hub`: For providing robust embedding models
- `sentence-transformers`: Text embeddings

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
* Streamlit – Interactive web app framework.
* LangChain – Framework for building LLM-powered apps.
* HuggingFace – For providing robust embedding models.
* FAISS – For lightning-fast similarity search.
* Google Gemini API – For powerful generative answers.
* Selenium & BeautifulSoup – For extracting structured data from dynamic and static web pages.
