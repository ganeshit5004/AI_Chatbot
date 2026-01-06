from fastapi import APIRouter, HTTPException, Depends
from app.schemas.models import (
    ChatRequest, 
    ChatResponse, 
    ChainCreateRequest,
    MemoryClearRequest
)
from app.core.config import settings
from typing import List, Dict
from app.core.langchain_instance import langchain_manager


router = APIRouter()



@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = langchain_manager.chat(
            chain_id=request.chain_id,
            message=request.message,
        )

        return ChatResponse(
            response=response,
            chain_id=request.chain_id,
            input_message=request.message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chains/create")
async def create_chain(request: ChainCreateRequest):
    """Create a new conversation chain"""
    try:
        langchain_manager.create_chain(
            chain_id=request.chain_id,
            system_prompt=request.system_prompt
        )
        return {"message": f"Chain '{request.chain_id}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chains")
async def list_chains():
    """List all active chains"""
    return {"chains": list(langchain_manager.chains.keys())}

@router.post("/memory/clear")
async def clear_memory(request: MemoryClearRequest):
    """Clear memory for a specific chain"""
    try:
        langchain_manager.clear_memory(request.chain_id)
        return {"message": f"Memory cleared for chain '{request.chain_id}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": settings.openai_model,
        "active_chains": len(langchain_manager.chains)
    }