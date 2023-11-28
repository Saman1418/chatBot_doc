from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os

# import urllib2

# proxy_support = urllib2.ProxyHandler({"http":"http://61.233.25.166:80"})
# opener = urllib2.build_opener(proxy_support)
# urllib2.install_opener(opener)

# html = urllib2.urlopen("http://www.google.com").read()
# print(html)

# import requests

# s = requests.Session()
# s.proxies = {"http": "http://61.233.25.166:80"}

# r = s.get("http://www.google.com")
# print(r.text)


# os.environ["OPENAI_API_KEY"] = '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'
# os.environ["OPENAI_API_KEY"] = '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'
os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom bot")


index = construct_index("docs")
iface.launch(share=True)