# pip install -q \
#   llama-index \
#   EbookLib \
#   html2text \
#   gradio \
#   llama-index-embeddings-huggingface \
#   llama-index-llms-ollama

import os
import textwrap
from pprint import pprint

# from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

os.chdir(r"D:\MyDrive2\pythonprojects\class\GENAI\LocalAILibrarian")
os.environ["PYTHONPATH"] = r"D:\MyDrive2\pythonprojects\ailib;" + os.environ.get("PYTHONPATH", "")
book_path = "books"
# GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# llm = Groq(api_key=GROQ_API_KEY)

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('BAAI/bge-small-en-v1.5')

# Function to check if model exists before initializing
# def check_and_initialize_embeddings(model_name):
#     model_path = os.path.expanduser(f"~\\.cache\\huggingface\\{model_name}")
#
#     # Check if the directory exists
#     if not os.path.exists(model_path):
#         print(f"Model path '{model_path}' does not exist. Please check the model name or ensure it is downloaded.")
#         return None
#
#     # Initialize embedding model
#     return HuggingFaceEmbedding(model_name=model_name)

llm_global = Ollama(model="phi3.5:latest", base_url="http://localhost:11434")
embed_model_global = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_folder=r"C:\Users\tropi\.cache\huggingface\hub")
# embed_model_global = check_and_initialize_embeddings("BAAI/bge-small-en-v1.5")
# if embed_model_global:
#     Settings.llm = llm_global
#     Settings.embed_model = embed_model_global
# else:
#     print("Failed to initialize embedding model due to missing files.")

Settings.llm = llm_global
Settings.embed_model = embed_model_global



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