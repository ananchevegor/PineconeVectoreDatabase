import os

from dotenv import load_dotenv
from pinecone_plugins.assistant.models.chat import Message

from uploader.uploadPDF import UploaderPDF
from pinecone import Pinecone

load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

pc_assist = Pinecone(api_key=os.getenv("PINECONE_ASSIST_API_KEY"))

index_name = "database-documents"
metadata_file = UploaderPDF("").get_metadata_from_file()

#text_file = UploaderPDF("").get_text_from_file()

#UploaderPDF("").insert_vectors_database(text_file, pc, index_name)


query = "Александр будет выполнять задачи связанные с дизайном"

# Search the dense index
dense_index = pc.Index(index_name)
results = dense_index.search(
    namespace=index_name,
    query={
        "top_k": 5,
        "inputs": {
            'text': query
        }
    })


"""
метрика для косинуса

Два почти идентичных предложения → 0.85 – 0.95

Синонимичные, но перефразированные → 0.75 – 0.85

Разные по смыслу → 0.4 – 0.6 или меньше

Порог отсечения часто устанавливают на 0.75–0.80

Для поиска по смыслу (semantic search) обычно используют top_k, а не только threshold

Для фильтрации дубликатов — используют > 0.90
"""

# Print the results
for hit in results['result']['hits']:
        print(
            f'id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | category: {hit['fields']['category']:<10} | text: {hit['fields']['text']:<50}')



