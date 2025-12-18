from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .api import router
from .lifespan import lifespan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable DEBUG logging for coordinator to see rejection reasons
logging.getLogger('idi.ian.coordinator').setLevel(logging.DEBUG)

app = FastAPI(
    title="IDI/IAN GUI Backend",
    description="Interface for Intelligent Daemon Interface",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "IDI/IAN Backend Online"}
