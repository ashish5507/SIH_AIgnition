# api.py

import uvicorn
import io
import asyncio
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, field_validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from src.predictor_proto import DeepSeaHybridClassifier
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
import pandas as pd
from typing import List, Dict

# --- 1. Production Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Application Setup ---
app = FastAPI(
    title="Deep Sea AI Backend (v11 - Production Grade)",
    version="11.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 2. Input Validation ---
class SequenceRequest(BaseModel):
    sequence: str

    @field_validator('sequence')
    def validate_dna(cls, v):
        invalid_chars = set(v.upper()) - set('ATGC')
        if invalid_chars:
            raise ValueError(f"Invalid DNA characters found: {', '.join(invalid_chars)}")
        if len(v) > 2000: # Add a max length check
            raise ValueError("Sequence length exceeds maximum of 2000 characters.")
        return v

# --- Model & Cache Loading ---
logging.info("Loading the DeepSeaHybridClassifier model...")
predictor = DeepSeaHybridClassifier()
logging.info("Model loaded successfully. API is ready.")

# --- 3. In-Memory BLAST Cache ---
blast_cache: Dict[str, str] = {}
# --- In-Memory storage for last result for downloading ---
last_result_cache: Dict = {}


# --- 4. Asynchronous BLAST with Timeout ---
async def run_blast_async(sequence: str) -> str:
    """Asynchronously runs BLAST with a cache and timeout."""
    if sequence in blast_cache:
        logging.info(f"Returning cached BLAST result for sequence: {sequence[:30]}...")
        return blast_cache[sequence]

    loop = asyncio.get_event_loop()
    try:
        logging.info(f"Running new BLAST query for sequence: {sequence[:30]}...")
        # Use asyncio.wait_for to add a timeout
        result_handle = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: NCBIWWW.qblast(program="blastn", database="nt", sequence=sequence, hitlist_size=1)
            ),
            timeout=90.0  # 90-second timeout
        )
        
        blast_record = NCBIXML.read(result_handle)
        result_handle.close()
        
        if blast_record.alignments:
            result = blast_record.alignments[0].title.split(' >')[0]
        else:
            result = "No significant similarity found."
        
        blast_cache[sequence] = result # Store result in cache
        return result
    except asyncio.TimeoutError:
        logging.error("BLAST query timed out.")
        return "BLAST query timed out after 90 seconds."
    except Exception as e:
        logging.error(f"Async BLAST query failed: {e}")
        return "BLAST query failed."


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Deep Sea AI Backend v11.0 is running."}

@app.post("/process_fasta")
async def process_fasta(
    file: UploadFile = File(...),
    top_n: int = Query(3, ge=1, le=5) # 5. Configurable Top-N Annotation
):
    try:
        contents = await file.read()
        fasta_string = contents.decode('utf-8').replace('\r\n', '\n').strip()
        records = list(SeqIO.parse(io.StringIO(fasta_string), "fasta"))
        if not records: raise ValueError("BioPython could not find valid FASTA records.")
            
        sequence_list = [str(record.seq) for record in records]
        
        predictions = predictor.predict_batch(sequence_list)
        
        df = pd.DataFrame(predictions)
        known_count = len(df[df['decision'] == 'ACCEPTED'])
        novel_count = len(df) - known_count
        
        for i, record in enumerate(records):
            predictions[i]['sequence_id'] = record.id or f"seq_{i+1}"

        abundance = df[df['decision'] == 'REJECTED']['final_label'].value_counts().nlargest(top_n)
        
        blast_tasks = []
        otu_labels_for_tasks = []
        for otu_label, _ in abundance.items():
            rep_idx = df[df['final_label'] == otu_label].index[0]
            rep_seq = sequence_list[rep_idx]
            blast_tasks.append(run_blast_async(rep_seq))
            otu_labels_for_tasks.append(otu_label)
        
        blast_results = await asyncio.gather(*blast_tasks)

        annotations = []
        for i, otu_label in enumerate(otu_labels_for_tasks):
            annotations.append({
                "otu_label": otu_label, "count": int(abundance[otu_label]), "top_blast_hit": blast_results[i]
            })

        global last_result_cache
        last_result_cache = {
            "summary_metrics": {"total_sequences": len(predictions), "known_taxa_detected": known_count, "novel_otus_detected": novel_count},
            "detailed_results": predictions, "annotation_report": annotations
        }
        logging.info("File processing complete.")
        return last_result_cache

    except Exception as e:
        logging.error(f"FATAL ERROR during file processing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

# --- 6. Downloadable Results Endpoint ---
@app.get("/download_csv")
def download_csv():
    global last_result_cache
    if not last_result_cache or 'detailed_results' not in last_result_cache:
        raise HTTPException(status_code=404, detail="No results available to download.")

    df = pd.DataFrame(last_result_cache['detailed_results'])
    
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=deep_sea_ai_results.csv"
    
    return response