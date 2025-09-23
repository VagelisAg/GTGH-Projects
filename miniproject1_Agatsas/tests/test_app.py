import pytest
from agatsas_miniproject1.app import SciFiExplorer 
import numpy as np

def test_len_documents():
    app=SciFiExplorer(data_directory="data")
    docs=app.load_documents()
    assert(len(docs)==5)

def test_text_splitter_chunks():
    app = SciFiExplorer(data_directory="data")
    docs = app.load_documents()
    chunks = app.text_splitter(docs)
    assert len(chunks) > 0


def test_vector_db_sets_retrievers():
    app = SciFiExplorer(data_directory="data")
    docs = app.load_documents()
    chunks=app.text_splitter(docs)
    vector_store,s_retriever,mmr_retriever=app.vector_DB(chunks)
    assert vector_store is not None
    assert s_retriever is not None
    assert mmr_retriever is not None









