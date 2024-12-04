# pip install -q \
#   llama-index \
#   EbookLib \
#   html2text \
#   gradio \
#   llama-index-embeddings-huggingface \
#   llama-index-llms-ollama

import os
import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index.llms.ollama import Ollama
import json
from pprint import pprint
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
# from llama_index.llms.groq import Groq
from groq import Groq
from llama_index.core import Settings
import textwrap
# from openai import OpenAI
from openai import ChatCompletion
# import openai
from llama_index.llms.openai import OpenAI

os.chdir(r"D:\MyDrive2\pythonprojects\class\GENAI\LocalAILibrarian")
# os.environ['TRANSFORMERS_CACHE'] = r'C:\Users\tropi\.cache\huggingface\hub'
book_path = "books"
# GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# llm_global = Groq(api_key=GROQ_API_KEY)
# llm_global = Ollama(model="phi3.5:latest", base_url="http://localhost:11434")
openai_api_key = os.getenv("OPENAI_API_KEY")
# llm_global = OpenAI(
#     # model_name="gpt-3.5-turbo",
#     base_url="",
#     api_key=openai_api_key
# )

# llm_global = ChatCompletion(
#     model='gpt-3.5-turbo',
#     api_key=openai_api_key
# )

llm_global = OpenAI(
    model="gpt-4o-mini",
    # api_key="some key",  # uses OPENAI_API_KEY env var by default
)
# client = OpenAI(
#     # model_name="gpt-3.5-turbo",
#     api_key=openai_api_key
# )

# llm_global = ChatCompletion(
#     model='gpt-3.5-turbo',
#     api_key=openai_api_key
# )

# llm_global = openai_model.Completion.create(model='gpt-3.5-turbo')
# llm_global = client.completions.create(model='gpt-3.5-turbo')
# llm_global = LLamaIndexOpenAI(
#     api_key=openai_api_key,
#     model_name="gpt-4o"  # Ensure this is the correct parameter for llama_index integration
# )

# embed_model_global = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embed_model_global = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    # model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="For this sentence, generate notation for use in retrieving related articles:"
)
# embed_model_global.query_instruction = "For this sentence, generate notation for use in retrieving related articles:"

Settings.llm = llm_global
Settings.embed_model = embed_model_global

# C:\Users\tropi\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5

class BookIndexer:
    def __init__(self,book_path):
        self.query_engine = None
        # self.summary_tool = None
        # self.vector_tool = None
        self.indexer = None
        self.docs_library = None
        self.book_path = book_path

    def read(self):
        self.docs_library = SimpleDirectoryReader(input_dir=self.book_path, required_exts=[".pdf", ".epub"]).load_data()

    # first half of RAG
    def do_index(self):
        self.indexer = VectorStoreIndex.from_documents(self.docs_library, embed_model=embed_model_global)
        # self.indexer.insert(
        #     self.docs_library,
        #     embed_model=embed_model_global,
        # )

    # second half of RAG
    def get_engines(self):
        vector_tool = QueryEngineTool(
            self.indexer.as_query_engine(llm=llm_global),
            metadata=ToolMetadata(
                name="vector_search",
                description="Useful for searching for specific facts.",
            ),
        )

        summary_tool = QueryEngineTool(
            self.indexer.as_query_engine(response_mode="tree_summarize"),
            metadata=ToolMetadata(
                name="summary",
                description="Useful for summarizing an entire document.",
            ),
        )

        self.query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool], select_multi=False, verbose=True)





# response = query_engine.query(
#     "Tell me about the specific details about Alan Turing's cryptographic methods at Bletchley Park"
# )

book_indexer = BookIndexer(book_path)
book_indexer.read()
book_indexer.do_index()
book_indexer.get_engines()


def Print(text, width=80, **args):
    lines = text.split('\n')  # Split the text into lines based on original line breaks
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=width))  # Wrap each line individually
    print('\n'.join(wrapped_lines), **args)

def GetSources(response):
  for node in response.source_nodes:
      # Access the TextNode object directly
      text_node = node.node

      # Assuming metadata is stored within the TextNode's metadata
      source = text_node.metadata.get('file_name') # Access metadata using .metadata.get()
      page = text_node.metadata.get('page_label')  # Access metadata using .metadata.get()

      Print(f"Source: {source[:30]}...")
      Print(f"Page: {page}")

def search(query):
    response = book_indexer.query_engine.query(query)
    GetSources(response)
    return response.response


pprint(search("Tell me about the specific details about Alan Turing's cryptographic methods at Bletchley Park"))


# def greet(fns:str,dt):
#     return f"Hello, {fns}, good {dt}!"
#
# with gr.Blocks() as demo:
#     gr.Markdown("# Greeting App")
#     output = gr.Textbox(label="Greeting")
#
#     with gr.Row():
#         first_name = gr.Textbox(label="Enter your first_name", placeholder="input your first name")
#         last_name = gr.Textbox(label="Enter your last_name", placeholder="input your last name")
#         afternoon = gr.Textbox(label="Enter the time", placeholder="input morning or afternoon")
#
#     fns = gr.Textbox(str(first_name.value) + str(last_name.value)+"ABC")
#
#     greet_btn = gr.Button("Greet")
#     greet_btn.click(fn=greet, inputs=(fns, afternoon),outputs=output)
#
# demo.launch()
