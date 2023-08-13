from fastapi import FastAPI, Request, Form
from transformers import pipeline
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# models = pipeline("summarization", model="nsi319/legal-pegasus")

tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")  
models = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus")

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
    input_tokenized = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors='pt',max_length=1024,truncation=True)
    summary_ids = models.generate(input_tokenized,
                                  num_beams=9,
                                  no_repeat_ngram_size=3,
                                  length_penalty=2.0,
                                  min_length=150,
                                  max_length=250,
                                  early_stopping=True)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]

    # Grab Currrent Time After Running the Code
    end = time.time()

    #Subtract Start Time from The End Time
    total_time = end - start
    print("\n"+ "TIME: " + str(total_time))

    return templates.TemplateResponse("result.html", {"request": request, "result": summary})

if __name__ == '__main__':
    uvicorn.run(app)
