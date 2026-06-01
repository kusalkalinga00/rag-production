import os
import tempfile
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader


from dotenv import load_dotenv

load_dotenv()


def load_text_file():

    # create a temporary text file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(b"This is a sample text file for testing.")
        temp_file_path = tmp_file.name

    try:
        # load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        for doc in documents:
            print(f"Document : {doc}")
            print(f"Loaded document content: {doc.page_content}")

    finally:
        # clean up the temporary file
        os.remove(temp_file_path)


def load_pdf_file(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents from PDF.")
    for i, doc in enumerate(documents):
        print(
            f"Document {i+1} content: {doc.page_content[:200]}..."
        )  # Print first 200 characters
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    # load_text_file()
    # Example PDF path (update with your actual PDF file path)
    pdf_path = "./docs/bitcoin.pdf"
    load_pdf_file(pdf_path)
