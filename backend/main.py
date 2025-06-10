from fastapi import FastAPI, UploadFile, File
import os

app = FastAPI()
UPLOAD_DIR = "/app/data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"status": "ok", "filename": file.filename}

@app.get("/")
def read_root():
    return {"message": "SDR decoder server is running"}
