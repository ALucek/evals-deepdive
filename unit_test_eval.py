import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Eval Unit Testing"

llama3 = ChatOllama(model='llama3', temperature=0)

prompt = PromptTemplate(
    template="""
    
    <|begin_of_text|>
    
    <|start_header_id|>system<|end_header_id|> 
    
    You are an AI assistant for generating python. Generate a python code snippet that answers the following question.
    Keep your answer concise, only include the necessary code snippet with no preamble or explanation.
    
    <|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>
    
    Question: {question} 
    Answer: 
    
    <|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

# Chain
chain = prompt | llama3 | StrOutputParser()

# Function
def generate(query):
    response = chain.invoke({'question': query})
    return response

out = generate("How do I write a for loop with range 10 in Python?")
print(out)