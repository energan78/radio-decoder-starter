# backend/web_interface.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="backend/static"), name="static")
templates = Jinja2Templates(directory="backend/templates")

SIGNAL_METADATA = "backend/signal_metadata.json"

@app.get("/viewer", response_class=HTMLResponse)
async def viewer(request: Request):
    with open(SIGNAL_METADATA) as f:
        signals = json.load(f)
    return templates.TemplateResponse("viewer.html", {"request": request, "signals": signals})

@app.get("/fft/{filename}")
async def fft_plot(filename: str):
    with open(SIGNAL_METADATA) as f:
        signals = json.load(f)
    match = next((s for s in signals if s["filename"] == filename), None)
    if not match:
        return HTMLResponse("<h2>Signal not found</h2>", status_code=404)

    iq = np.fromfile(match["path"], dtype=np.complex64)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq[:1024])))
    plt.figure(figsize=(8, 4))
    plt.plot(spectrum)
    plt.title(f"FFT of {filename}")
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude")
    fft_img = f"backend/static/{filename}.png"
    plt.savefig(fft_img)
    plt.close()
    return FileResponse(fft_img)

@app.post("/update")
async def update_comment(filename: str = Form(...), new_class: str = Form(...), comment: str = Form(...)):
    with open(SIGNAL_METADATA) as f:
        signals = json.load(f)
    for s in signals:
        if s["filename"] == filename:
            s["class"] = new_class
            s["comment"] = comment
    with open(SIGNAL_METADATA, "w") as f:
        json.dump(signals, f, indent=2)
    return RedirectResponse("/viewer", status_code=303)
