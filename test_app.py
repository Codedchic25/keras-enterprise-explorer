import pytest
import numpy as np
import keras
from app import respond_chat_rag, dataset, all_vectors

def test_dataset_loading():
    assert len(dataset.documents) > 0

def test_rag_successful_match():
    response = respond_chat_rag("mixed precision", history=[])
    assert "Telemetry:" in response
    assert "Nu am găsit reguli" not in response

def test_rag_fallback_threshold():
    response = respond_chat_rag("text complet aleatoriu fara legatura cu proiectul", history=[])
    assert "Nu am găsit reguli enterprise" in response