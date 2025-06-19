import os
import time
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from PIL import Image
from sentence_transformers import SentenceTransformer
import uploader.uploadFiles
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

""" Загрузка ТЕКСТА на КАРТИНКЕ в базу данные"""

"""uploader.uploadFiles.FileUploader("").define_image_as_vector(index)"""


"""client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Where is Europe?"
)
embedding = response.data[0].embedding


results = index.query(
    namespace=index_name,
    vector=embedding,
    top_k=2,
    include_metadata=True  # Чтобы получить текст и категорию
)

print(results)"""



image_name = "images/jupiter.jpg"
index_name = "database-pictures"

uploaderImage = uploader.uploadFiles.FileUploader(image_name)
image_embedding = uploaderImage.define_image_as_vector(image_name)

index = pc.Index(index_name)

#uploaderImage.upload_image_as_vector(image_embedding[0], index, "")

stats = uploaderImage.get_image_vector_db(index, "", image_embedding[0])
print(stats)
