import os

# 1. Configurare Hardware Enterprise
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

import gradio as gr
import numpy as np
import keras
import keras.ops as ops
from data_pipeline import KerasEnterpriseDataset, compute_keras_similarity

COSINE_THRESHOLD = 0.10

# Încărcare dataset enterprise
dataset = KerasEnterpriseDataset(json_path="mock_docs.json", batch_size=32)
all_docs = dataset.documents
all_vectors = dataset.get_all_vectors_optimized()

# Mapare deterministă internă pentru toate cele 13 reguli enterprise
DETERMINISTIC_RESPONSES = {
    "keras_rule_01": "Folosește exclusiv namespace-ul unificat `keras.ops` pentru a menține compatibilitatea totală multi-backend (JAX, TensorFlow, PyTorch).",
    "keras_rule_02": "Utilizează clasa `keras.utils.PyDataset`, care înlocuiește vechea clasă `Sequence` pentru I/O asincron din Python și performanță maximă.",
    "keras_rule_03": "Activează politica globală folosind comanda: `keras.mixed_precision.set_global_policy('mixed_float16')`.",
    "keras_rule_04": "În Keras 3, funcțiile straturilor personalizate trebuie să fie stateless, să accepte intrări curate și să nu modifice direct stările tensorilor interni.",
    "keras_rule_05": "Salvează modelele de producție folosind formatul standard modern `.keras` prin comanda: `model.save('model.keras')`.",
    "keras_rule_06": "Toate funcțiile custom de loss și metrici trebuie scrise cu `keras.ops`. Folosirea NumPy sau a tensorilor nativi va strica graful de compilare.",
    "keras_rule_07": "Optimizează modelul pentru hardware prin activarea compilării Just-In-Time (JIT) adăugând argumentul `jit_compile=True` în `model.compile()`.",
    "keras_rule_08": "Inițializează distribuția cu `strategy = keras.distribution.MirroredDistribution()` și construiește modelul în blocul `with strategy.scope():`.",
    "keras_rule_09": "Moștenește clasa `keras.optimizers.schedules.LearningRateSchedule`. Nu modifica rata de învățare manual în loop, pentru că strici graful.",
    "keras_rule_10": "Creează un callback custom pe metoda `on_epoch_end` care rulează `gc.collect()` și `keras.backend.clear_session()` pentru a preveni crash-urile OOM.",
    "keras_rule_11": "Nu utiliza blocuri python native 'if/else' în codul compilat. Folosește operatorul simbolic grafic dedicat: `keras.ops.cond()`.",
    "keras_rule_12": "Subclasează structura API `keras.quantizers.Quantizer` pentru a converti greutățile rețelei în format comprimat int8 pe dispozitive Edge.",
    "keras_rule_13": "Pentru modele LLM gigantice, configurează un `keras.distribution.DeviceMesh` și utilizează specificații de tip Layout pentru Sharding automat.",
}

# CSS Premium pentru container centrat și culori închise enterprise
CUSTOM_CSS = """
.gradio-container { background-color: #0b0f19 !important; font-family: 'Segoe UI', system-ui, sans-serif !important; max-width: 850px !important; margin: 0 auto !important; padding: 25px !important; border: 1px solid #1f2937 !important; border-radius: 16px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
.enterprise-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #1e1b4b, #0f172a); border-radius: 12px; margin-bottom: 25px; border: 1px solid #312e81; }
.telemetry-container { background-color: #111827; border: 1px solid #1f2937; padding: 12px 15px; border-radius: 8px; font-size: 13px; color: #9ca3af; margin-top: 15px; font-family: monospace; display: block; line-height: 1.5; }
"""

def respond_chat_rag(message, history):
    """Procesează mesajele utilizând formatul universal de liste clasice compatible local."""
    if history is None:
        history = []

    try:
        query_vector = dataset._text_to_vector(message)
        scores = compute_keras_similarity(query_vector, all_vectors)
        scores_numpy = ops.convert_to_numpy(scores).flatten()

        best_match_idx = int(np.argmax(scores_numpy))
        highest_score = float(scores_numpy[best_match_idx])
        matched_doc = all_docs[best_match_idx]

        if highest_score < COSINE_THRESHOLD:
            answer = "❌ Nu am găsit reguli enterprise sau documentație relevantă pentru această interogare în baza de date locală."
            telemetry = f"⚠️ Respins sub pragul critic | Cosine: {highest_score:.4f} | Backend: {keras.config.backend()}"
        else:
            answer = DETERMINISTIC_RESPONSES.get(matched_doc["id"], "ℹ️ Regulă identificată fără text determinist.")
            telemetry = f"📄 {matched_doc['title']} | Cosine: {highest_score:.4f} | Backend: {keras.config.backend()}"

        full_response = f"{answer}\n\n📊 Telemetry:\n{telemetry}"
        return full_response
    except Exception as e:
        return f"⚠️ Eroare tehnică: {str(e)}"

# Blocurile de interfață cu tema integrată global
with gr.Blocks() as demo:
    gr.HTML(
        "<div class='enterprise-header'>"
        "<h1 style='color:#ffffff; margin-bottom:5px; font-size:22px;'>🚀 Keras Enterprise Core v3</h1>"
        "<p style='color:#9ca3af; font-size:13px;'>Neural Chat Interface & Hybrid Matrix Similarity Engine (Keras Ops / TF-IDF)</p>"
        "</div>"
    )
    gr.ChatInterface(fn=respond_chat_rag)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS)
