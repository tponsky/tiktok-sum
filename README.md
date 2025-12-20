# VidRecall

A simple RAG (Retrieval-Augmented Generation) application that allows you to search through your video library using natural language.

## Prerequisites

- Python 3.8+
- OpenAI API Key
- Pinecone API Key & Index

## Setup

1.  **Clone/Download** the repository.
2.  **Install Dependencies**:
    ```bash
    pip install fastapi uvicorn openai pinecone python-dotenv
    ```
3.  **Environment Variables**:
    Create a `.env` file in the root directory with your API keys:
    ```env
    OPENAI_API_KEY=your_openai_key
    PINECONE_API_KEY=your_pinecone_key
    ```
    *Note: The Pinecone index name is hardcoded as `tiktok-rag` in `rag_api.py`. Change it there if needed.*

## Running the App

1.  **Start the Server**:
    ```bash
    uvicorn rag_api:app --reload
    ```
2.  **Access the App**:
    Open your browser and go to:
    [http://localhost:8000](http://localhost:8000)

## Usage

1.  **Search**: Type a question or topic into the search bar (e.g., "healthy dinner recipes").
2.  **View Results**:
    - **AI Answer**: A summarized answer based on the retrieved video content.
    - **Source Videos**: A list of relevant videos with links to watch them on TikTok.

## API Endpoints

-   `GET /`: Serves the search interface.
-   `GET /search?q=...`: Search endpoint (returns JSON).
-   `POST /search`: Search endpoint (accepts JSON body).

## Adding Videos (Ingestion)

To add new TikTok videos to your knowledge base, use the `tiktok_rag_cloud.py` script.

### Prerequisites
-   `ffmpeg` must be installed on your system.
-   Python dependencies installed (see Setup).

### Usage

1.  **Copy a TikTok URL** to your clipboard.
2.  Run the script:
    ```bash
    python tiktok_rag_cloud.py
    ```
    It will automatically detect the URL from your clipboard, download, transcribe, summarize, and add it to the index.

3.  **Alternatively**, pass URLs directly:
    ```bash
    python tiktok_rag_cloud.py https://www.tiktok.com/@user/video/1234567890
    ```
