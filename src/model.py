from transformers import pipeline

qa_pipeline = pipeline("question-answering")

def generate_answer(context, question):
    return qa_pipeline({'context': context, 'question': question})['answer']
