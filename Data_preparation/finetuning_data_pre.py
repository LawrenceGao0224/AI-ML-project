# Install package
# !pip install pyMuPDF
# !pip install ollama


# Extract Data From PDF------------------------------------------------------
import fitz
"""
Extract each page of the text and retrun it as the list.
pdf_path: path of the pdf file
return: list of string
"""
def extract_data_from_pdf(pdf_path):
  # open document
  document = fitz.open(pdf_path)
  text_list = []
  for page in document:
    text = page.get_text()
    text_list.append(text)
  document.close()
  
  return text_list

pdf_path = "C:/Users/kaosh/Desktop/All_project/Side-project-ML/github/Data_preparation/Terraform Training - Module 1.pdf"

text_data = extract_data_from_pdf(pdf_path)


# Parsing Function----------------------------------------------------------------------------------------------------------------

QA_list = []
counter = 0
failed_gen_counter = 0

import json

def parse_coding_question_data(data, failed_gen):
  try:
    # Split the input part into Question and Answer 
    parts = data.strip().split("**Answer:**")
    question = parts[0].strip()
    answer = parts[1].strip()

    # Extract the question text
    question_text = question.part.replace("**Question:**", "").strip()
    
    # Create jason object
    question_json = {
        "user": question_text,
        "assistant": answer
    }
    
    # Store in a list
    question_list = [question_json]

    return question_list
  except Exception as e:
    failed_gen += 1
    pass


# Inference Loop----------------------------------------------------------------------------------------------------------------------------
import ollama

for page_number, page_text in enumerate(text_data):
  context = page_text

  response = ollama.chat(model="llama2", messages=[
      {"role": "system", "content": "You are a helpful assistant to make up a coding question and answer."},
      {"role": "user", "content": f"Make up a coding question and answer based on the context{context}, and have a standardize Sections like **Questions** and **Answer**",}
  ])
  result = parse_coding_question_data(response['message']['content'], failed_gen_counter)
  print(response['message']['content'])
  print("_______________________")

  if result is not None:
    QA_list.append(result[0]) 
    print(result[0])
    print("__________________")

  counter += 1

  if counter >= 2:
    break
  


# Save into json file----------------------------------------------------------------------------------------------------------------------------
file_path = "C:/Users/kaosh/Desktop/All_project/Side-project-ML/github/Data_preparation/data.json"

with open(file_path, "w") as file:
    for item in QA_list:
      json_line = json.dump(item)
      file.write(json_line+ "\n")