from fastapi import FastAPI, Request, Form
from transformers import pipeline
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

models = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/form", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/summarize", response_class=HTMLResponse)
def summarize(request: Request, ARTICLE_TO_SUMMARIZE: str = Form(...)):
    """
    A simple function that receive a text and summarize it.
    """
    # Grab Currrent Time Before Running the Code
    start = time.time()

    result = models(ARTICLE_TO_SUMMARIZE, min_length=30, do_sample=False)

    summary = result[0]["summary_text"]

    # Grab Currrent Time After Running the Code
    end = time.time()

    #Subtract Start Time from The End Time
    total_time = end - start
    print("\n"+ "TIME: " + str(total_time))

    return templates.TemplateResponse("result.html", {"request": request, "result": summary})

if __name__ == '__main__':
    uvicorn.run(app)
