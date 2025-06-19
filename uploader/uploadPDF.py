import random
import time

import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
import fitz
from unicodedata import category


class UploaderPDF:
    def __init__(self, filename):
        self.filename = filename


    def get_metadata_from_file(self):
        doc = fitz.open(self.filename)
        print(doc.metadata)


    def get_text_from_file(self):
        md_text = pymupdf4llm.to_markdown(self.filename)#
        splitter = MarkdownTextSplitter(chunk_size=250, chunk_overlap=0)
        md_array = splitter.create_documents([md_text]) # это уже подготовленные чанки, которые должен будут пойти в базу
        # md_array.page_content - это чанк разбитый на 250 символов
        return md_array

    def insert_vectors_database(self, chanks_array, pc, index_name):
            records = []
            for chunk in chanks_array:
                chunk_id =  random.randint(100000000, 999999999)
                records.append({"_id": str(chunk_id), "text": str(chunk.page_content.replace('\n', ' ').strip()), "category": "document"})
            print(records)
            try:

                dense_index = pc.Index(index_name)
                dense_index.upsert_records(index_name, records)

                time.sleep(10)
                stats = dense_index.describe_index_stats()
                print(stats)

            except Exception as e:
                print(e)
            


