from fastapi import FastAPI
from pydantic import BaseModel
import yaml
from utils.rag_pipeline import retrieve_chunks, build_messages
from utils.llm_providers import chat as llm_chat

app = FastAPI(title="Industrial QA RAG API", version="0.1.0")
CFG = yaml.safe_load(open('configs/llm.yaml','r'))

class ChatReq(BaseModel):
    question: str
    top_k: int = CFG['retrieval']['top_k']
    max_ctx: int = CFG['retrieval']['max_context_chars']

@app.post("/chat")
def chat(req: ChatReq):
    hits, _ = retrieve_chunks(req.question, "vectorstore", top_k=req.top_k)
    messages = build_messages(req.question, hits, max_chars=req.max_ctx)
    out = llm_chat(CFG['provider'], messages, CFG)
    return {"answer": out, "sources": hits}
