from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from io import BytesIO
from PIL import Image
import uvicorn

app = FastAPI()

@app.post("/vqa")
async def vqa_endpoint(question: str = Form(...), image_0: UploadFile | None = File(None),
                       image_1: UploadFile | None = File(None), image_2: UploadFile | None = File(None), image_3: UploadFile | None = File(None)):
    imgs: List[Image.Image] = []
    for uf in [image_0, image_1, image_2, image_3]:
        if uf is None: continue
        data = await uf.read()
        try: imgs.append(Image.open(BytesIO(data)).convert("RGB"))
        except Exception: pass
    # TODO: replace with real model call
    return JSONResponse({"answer": "none", "scores": {"vaping":0.01, "smoking":0.01, "none":0.98}})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)