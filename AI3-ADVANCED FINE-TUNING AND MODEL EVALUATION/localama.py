from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import warnings
warnings.filterwarnings("ignore")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Your current task is to perform summarization which should contain necessary information present in the text. You have to create a concise good summary. make sure that that summary should contain  in one line"),
        ("user","Question:{question}")
    ]
)
input_text = "Today various organizations, be it online shopping, government and private sector organizations, catering and tourism industry or other institutions that offer customer services are concerned about their customers and ask for feedback every single time we use their services. Consider the fact, that these companies may be receiving enormous amounts of user feedback every single day. And it would become quite tedious for the management to sit and analyze each of those."

llm=Ollama(model="qwen2.5-coder:0.5b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

answer = chain.invoke({"question":input_text})
print(answer)