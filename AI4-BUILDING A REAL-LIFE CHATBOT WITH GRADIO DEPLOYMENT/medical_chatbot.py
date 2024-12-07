import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Step 1: Define the prompt template and the chain

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a health care assistant bot. You have to guide and answer to the user queries regarding any medical problems, medicines, healthcare guidelines,medicine guidelines, etc."),
        ("user", "Question:{question}")
    ]
)

# Define the input text
def health_assistant_bot(question: str):
    # Step 2: Define the LLM and output parser
    llm = Ollama(model="phi:2.7b-chat-v2-q2_K")
    output_parser = StrOutputParser()
    
    # Step 3: Create the chain
    chain = prompt | llm | output_parser

    # Step 4: Invoke the chain with user input
    answer = chain.invoke({"question": question})
    return answer

# Step 5: Create Gradio interface

iface = gr.Interface(
    fn=health_assistant_bot,  # function to be called
    inputs=gr.Textbox(label="Ask a healthcare question"),  # input field
    outputs=gr.Textbox(label="Answer"),  # output field
    title="Healthcare Assistant Bot",  # Title of the UI
    description="Ask healthcare-related questions, and the assistant will provide guidance on medical problems, medicines, and healthcare guidelines."
)

# Step 6: Launch the interface
iface.launch()
