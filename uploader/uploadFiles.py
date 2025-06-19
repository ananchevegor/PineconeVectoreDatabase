import os
import random
import uuid
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()


class FileUploader:
    def __init__(self, filename):
        self.filename = filename

    def define_text_from_image_as_vector(self, index_name):
        text_from_image = self.get_text_from_image()
        get_chunks = self.get_chunks_from_image_text(text_from_image)
        print(get_chunks)
        for i, chunk in enumerate(get_chunks):
            vector = self.get_embedding_from_open_ai(chunk)
            print(vector)
            index_name.upsert(
                vectors=[
                    {
                        "id": f"v_ch_{random.randint(1000, 9999)}",
                        "values": vector,
                        "metadata": {
                            "text": chunk,
                            "source": self.filename
                        }
                    }
                ],
                namespace="database-images"
            )

    def get_text_from_image(self):
        """ Взять текст из картинки(OCR)"""
        image = Image.open(self.filename)
        image = image.convert("L")
        version = pytesseract.get_tesseract_version()
        print("Tesseract версия:", version)
        text = pytesseract.image_to_string(image, lang="eng")
        return text

    @staticmethod
    def get_chunks_from_image_text(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        return chunks

    @staticmethod
    def get_embedding_from_open_ai(text):
        client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    @staticmethod
    def define_image_as_vector(self, image):
        model = SentenceTransformer("clip-ViT-B-32")
        image = Image.open(image)
        embedding = model.encode([image], convert_to_tensor=False)
        embedding_vector = embedding[0]
        return embedding_vector, embedding.shape[1]


    def upload_image_as_vector(self, image_vector, index, namespace):
        index.upsert([
            {
                "id": str(uuid.uuid4()),
                "values": image_vector,
                "metadata": {"type": "image", "source": self.filename},

            }
        ], namespace=namespace)

    @staticmethod
    def get_image_vector_db(index, namespace, image_vector):
        result = index.query(
            namespace=namespace,
            vector=image_vector,
            top_k=3,
            include_metadata=True,
            include_values=False
        )
        return result
