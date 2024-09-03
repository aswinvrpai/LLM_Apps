# LangChain Expression Language
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

import dotenv
dotenv.load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# LLM;
llm_model=ChatGroq(model="Gemma2-9b-It",groq_api_key=GROQ_API_KEY)
print(llm_model)

# Using Chat Prompt Templates;
generic_template="Translate following into {language}"

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",generic_template),
        ("user","{text}")
    ]
)

# Parser;
parser=StrOutputParser()

# Chain
chain=prompt|llm_model|parser
# result=chain.invoke({"language":"Japanese","text":"How are you?"})

# Fast API app;
app=FastAPI(
    title="LLM Demo App",
    description="An interface of LLM App",
    version="1.0"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port="8000")



