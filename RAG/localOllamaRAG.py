import requests
import numpy as np
import json

# Replace 'your-ollama-url' with your Ollama server URL
OLLAMA_URL = 'http://127.0.0.1:11434/api/generate'
model = 'llama3.1:latest' # TODO: update this for whatever model you wish to use

def get_text(my_file):
    # reading the file 
    data = my_file.read() 
    # replacing end splitting the text  
    # when newline ('\n') is seen. 
    data_into_list = data.split("\n") 
    my_file.close() 

    return data_into_list


def retrieve_documents(query, documents, top_k=3):
    # A simple retrieval mechanism using cosine similarity
    query_vector = np.array(query_to_vector(query))
    scores = [(doc, cosine_similarity(query_vector, doc_vector(doc))) for doc in documents]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scores[:top_k]]

def query_to_vector(query):
    # Placeholder for converting a query to a vector
    # Replace with your actual implementation
    return np.random.rand(512)

def doc_vector(doc):
    # Placeholder for converting a document to a vector
    # Replace with your actual implementation
    return np.random.rand(512)

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def generate(prompt):
    r = requests.post(OLLAMA_URL,
                      json={
                          'model': model,
                          'prompt': prompt,
                      },
                      stream=True)
    r.raise_for_status()

    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        # the response streams one token at a time, print that as we receive it
        print(response_part, end='', flush=True)

        if 'error' in body:
            raise Exception(body['error'])

        if body.get('done', False):
            return body['context']

def main():

    my_file = open('C:/Users/kaosh/Desktop/All_project/Side-project-ML/github/RAG/About_Lawrence.txt','r')
    documents = get_text(my_file)

    while True:
        user_input = input("Enter your question: ")
        retrieved_docs = retrieve_documents(user_input, documents)
        context = "\n".join(retrieved_docs)
        
        if not user_input:
            exit()
        print()
        
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
        context = generate(prompt)
        print()

if __name__ == "__main__":
    main()