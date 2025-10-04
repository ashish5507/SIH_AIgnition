import uvicorn
import io
import asyncio
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, field_validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from src.predictor_proto import DeepSeaHybridClassifier
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
import pandas as pd
from typing import List, Dict

# --- Logging and App Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(title="Deep Sea AI Backend", version="12.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Input Validation Models ---
class DNASequence(BaseModel):
    sequence: str
    @field_validator('sequence')
    def validate_dna(cls, v):
        invalid_chars = set(v.upper()) - set('ATGC')
        if invalid_chars:
            raise ValueError(f"Invalid DNA characters found: {', '.join(invalid_chars)}")
        if len(v) > 2000:
            raise ValueError("Sequence length exceeds maximum of 2000 characters.")
        return v

# --- Model & Cache (Lazy init) ---
predictor: DeepSeaHybridClassifier = None
blast_cache: Dict[str, str] = {}
last_result_cache: Dict = {}

def get_predictor() -> DeepSeaHybridClassifier:
    """Ensure predictor is initialized only once (lazy)."""
    global predictor
    if predictor is None:
        logging.info("Lazy loading DeepSeaHybridClassifier...")
        predictor = DeepSeaHybridClassifier()
        logging.info("Predictor initialized successfully.")
    return predictor

# --- Asynchronous BLAST ---
async def run_blast_async(sequence: str) -> str:
    if sequence in blast_cache:
        return blast_cache[sequence]
    loop = asyncio.get_event_loop()
    try:
        result_handle = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: NCBIWWW.qblast("blastn", "nt", sequence, hitlist_size=1)),
            90.0
        )
        blast_record = NCBIXML.read(result_handle)
        result_handle.close()
        result = blast_record.alignments[0].title.split(' >')[0] if blast_record.alignments else "No significant similarity found."
        blast_cache[sequence] = result
        return result
    except asyncio.TimeoutError:
        return "BLAST query timed out after 90 seconds."
    except Exception as e:
        return f"BLAST query failed: {e}"

# --- API Endpoints ---
@app.get("/")
def serve_html_frontend():
    """Serves the main index.html file to the user."""
    return FileResponse('index.html')

# ==============================================================================
# === ENDPOINT FOR SINGLE SEQUENCE PREDICTION ===
# ==============================================================================
@app.post("/predict_single")
def predict_single(item: DNASequence):
    """Receives a single DNA sequence string and returns its classification."""
    try:
        logging.info(f"Received single sequence request for: {item.sequence[:30]}...")
        predictor_instance = get_predictor()
        predictions = predictor_instance.predict_batch([item.sequence])
        if not predictions:
            raise HTTPException(status_code=500, detail="Model returned no prediction.")
        return predictions[0]
    except Exception as e:
        logging.error(f"Error during single sequence prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_fasta")
async def process_fasta(file: UploadFile = File(...), top_n: int = Query(3, ge=1, le=5)):
    try:
        contents = await file.read()
        fasta_string = contents.decode('utf-8').strip()
        records = list(SeqIO.parse(io.StringIO(fasta_string), "fasta"))
        if not records:
            raise ValueError("No valid FASTA records found.")

        predictor_instance = get_predictor()
        sequence_list = [str(r.seq) for r in records]
        predictions = predictor_instance.predict_batch(sequence_list)

        df = pd.DataFrame(predictions)
        known_count = len(df[df['decision'] == 'ACCEPTED'])
        novel_count = len(df) - known_count

        for i, r in enumerate(records):
            predictions[i]['sequence_id'] = r.id or f"seq_{i+1}"

        abundance = df[df['decision'] == 'REJECTED']['final_label'].value_counts().nlargest(top_n)
        blast_tasks, otu_labels = [], []
        for label, _ in abundance.items():
            rep_idx = df[df['final_label'] == label].index[0]
            rep_seq = sequence_list[rep_idx]
            blast_tasks.append(run_blast_async(rep_seq))
            otu_labels.append(label)

        blast_results = await asyncio.gather(*blast_tasks)
        annotations = []
        for i, label in enumerate(otu_labels):
            annotations.append({
                "otu_label": label,
                "count": int(abundance[label]),
                "top_blast_hit": blast_results[i]
            })

        global last_result_cache
        last_result_cache = {
            "summary_metrics": {
                "total_sequences": len(predictions),
                "known_taxa_detected": known_count,
                "novel_otus_detected": novel_count
            },
            "detailed_results": predictions,
            "annotation_report": annotations
        }
        return last_result_cache
    except Exception as e:
        logging.error(f"FATAL ERROR during file processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_csv")
def download_csv():
    global last_result_cache
    if not last_result_cache:
        raise HTTPException(404, "No results available.")
    df = pd.DataFrame(last_result_cache['detailed_results'])
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=deep_sea_ai_results.csv"
    return response