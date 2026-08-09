from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# Load vectorstore from the rag folder next to this script
VECTORSTORE_PATH = Path(__file__).resolve().parent / "vectorstore"


# ==========================================
# Load embedding model
# ==========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=768,
)


# ==========================================
# Load FAISS database
# ==========================================

vectorstore = FAISS.load_local(
    str(VECTORSTORE_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)


# ==========================================
# Create retriever
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)


def retrieve_documents(query: str):

    documents = retriever.invoke(query)

    return documents


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    query = "How can shaft misalignment in a centrifugal pump be diagnosed?"

    documents = retrieve_documents(query)

    print("\nQUERY:")
    print(query)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, document in enumerate(documents, start=1):

        print(f"\n--- Result {i} ---")

        print("Source:", document.metadata.get("source"))

        print("Content:")
        print(document.page_content)