from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================================
# Paths
# ==========================================

DOCUMENTS_PATH = Path(__file__).resolve().parent.parent / "documents"
# Place vectorstore inside the rag folder so index files live at rag/vectorstore/
VECTORSTORE_PATH = Path(__file__).resolve().parent / "vectorstore"


# If DOCUMENTS_PATH doesn't exist, show helpful message and exit
if not DOCUMENTS_PATH.exists():
    raise FileNotFoundError(f"Documents directory not found: {DOCUMENTS_PATH}")
# Ensure the vectorstore directory exists
VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# Load documents
# ==========================================

documents = []

for file_path in DOCUMENTS_PATH.glob("*.txt"):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file_path.name

    documents.extend(docs)


print(f"Loaded {len(documents)} documents")

if len(documents) == 0:
    print("No documents found to ingest. Put .txt files into the documents directory and retry.")
    raise SystemExit(1)


# ==========================================
# Split documents into chunks
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

if len(chunks) == 0:
    print("No chunks were created from documents. Check the splitter settings or document contents.")
    raise SystemExit(1)


# ==========================================
# Create embeddings
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=768,
)


# ==========================================
# Create FAISS vector database
# ==========================================

vectorstore = FAISS.from_documents(chunks, embeddings)


# ==========================================
# Save vector database
# ==========================================

vectorstore.save_local(str(VECTORSTORE_PATH))

print("FAISS vector database created successfully!")