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
from pprint import pprint
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine
# from llama_index.llms.groq import Groq
from llama_index.core import Settings
import textwrap
import platform
import shutil

# GROQ_API_KEY = os.environ["GROQ_API_KEY"]
# llm_global = Groq(api_key=GROQ_API_KEY)
from llama_index.llms.ollama import Ollama
llm_global = Ollama(model="phi3.5:latest", base_url="http://localhost:11434")
from llama_index.llms.openai import OpenAI
# openai_api_key = os.getenv("OPENAI_API_KEY")
# llm_global = OpenAI(
#     model="gpt-4o-mini",
#     api_key=openai_api_key,  # uses OPENAI_API_KEY env var by default
# )



embed_model_global = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# embed_model_global = HuggingFaceBgeEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",
#     # model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
#     query_instruction="For this sentence, generate notation for use in retrieving related articles:"
# )
# embed_model_global.query_instruction = "For this sentence, generate notation for use in retrieving related articles:"

# Settings.llm = llm_global
# Settings.embed_model = embed_model_global

class BookIndexer:
    def __init__(self,book_path):
        self.query_engine = None
        self.indexer = None
        self.docs_library = None
        self.book_path = book_path

    def read(self):
        self.docs_library = SimpleDirectoryReader(input_dir=self.book_path, required_exts=[".pdf", ".epub"]).load_data()

    # first half of RAG
    def do_index(self):
        self.indexer = VectorStoreIndex.from_documents(self.docs_library, embed_model=embed_model_global)

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

    def prepare_query(self):
        self.read()
        self.do_index()
        self.get_engines()

    def query(self,querystr:str):
        result = self.query_engine.query(querystr)
        return result

class query_library_helper:
    def Print(self,text, width=80, **args):
        lines = text.split('\n')  # Split the text into lines based on original line breaks
        wrapped_lines = []
        for line in lines:
            wrapped_lines.extend(textwrap.wrap(line, width=width))  # Wrap each line individually
        print('\n'.join(wrapped_lines), **args)

    def GetSources(self,response):
      for node in response.source_nodes:
          # Access the TextNode object directly
          text_node = node.node

          # Assuming metadata is stored within the TextNode's metadata
          source = text_node.metadata.get('file_name') # Access metadata using .metadata.get()
          page = text_node.metadata.get('page_label')  # Access metadata using .metadata.get()

          self.Print(f"Source: {source[:30]}...")
          self.Print(f"Page: {page}")

    def search(self,query_engine, query):
        response = query_engine.query(query)
        self.GetSources(response)
        return response.response

    def clear_path(self, folderpath):
        try:
            # List all entries in the directory given by folder_path
            for entry in os.scandir(folderpath):
                os.unlink(entry.path)
                print(f"Deleted symlink: {entry.path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def clear_soft_link(self, folderpath):
        try:
            # List all entries in the directory given by folder_path
            for entry in os.scandir(folderpath):
                # Check if the entry is a symbolic link
                if entry.is_symlink():
                    # Delete the symbolic link
                    os.unlink(entry.path)
                    print(f"Deleted symlink: {entry.path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_os_type(self):
        os_type = platform.system()
        if os_type == "Linux":
            return "Linux"
        elif os_type == "Windows":
            return "Windows"
        else:
            return f"Unknown OS: {os_type}"

class LibraryInterface:
    def __init__(self, libpath:str) -> None:
        self.file_list = []
        self.library_path = libpath
        self.qhelper = query_library_helper()
        self.qhelper.clear_path(self.library_path)

    def include_file_to_list(self, filepath:str)->str:
        self.file_list.append(filepath)
        return str(self.file_list)

    def query_library(self, query:str)->str:
        if len(query)==0:
            return "Please enter your query"
        if len(self.file_list)==0:
            return "Please add books first"

        self.put_book_into_folder()
        book_indexer = BookIndexer(self.library_path)
        book_indexer.prepare_query()
        response = book_indexer.query(query)
        self.qhelper.GetSources(response)
        return response.response

    def put_book_into_folder(self)->None:
        ostype = self.qhelper.get_os_type()

        #remove redundent
        unique_set = set(self.file_list)

        for i in unique_set:
            if os.path.exists(i):
                if ostype == "Windows":
                    shutil.copy2(i, os.path.join(self.library_path, os.path.basename(i)))
                else:
                    os.symlink(i, os.path.join(self.library_path, os.path.basename(i)))

    def displayui(self):
        with gr.Blocks() as demo:
            with gr.Row():
                file_ctl = gr.File(label="Select book")
                lab_list = gr.Textbox(label="Library")
                with gr.Column():
                    add_ctl = gr.Button("Add book")
                    query_ctl = gr.Button("Search")
            query_input = gr.Textbox(label="Input your query", value="what these two books about?")
            query_output = gr.Textbox(label="RAG output")
            add_ctl.click(fn=self.include_file_to_list, inputs=file_ctl, outputs=lab_list)
            query_ctl.click(fn=self.query_library, inputs=query_input, outputs=query_output)
        demo.launch()

def main():
    os.chdir(r"D:\MyDrive2\pythonprojects\class\GENAI\LocalAILibrarian")
    book_path = "books"
    book_path_abs = os.path.abspath(book_path)
    # pprint(search(book_indexer, "Tell me about the specific details about Alan Turing's cryptographic methods at Bletchley Park"))
    li = LibraryInterface(book_path_abs)
    li.displayui()

if __name__ == "__main__":
    main()