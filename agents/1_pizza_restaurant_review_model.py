from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectory import retriver

model = OllamaLLM(model="llama3.2")
template = """
You are an expert in pizza restaurants

Here are some relevant reviews: {reviews}.

Here is the question to answer: {question}.

"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n")
    user_question = input("Ask a question about pizza restaurants(q to quit): ")
    if user_question == "q":
        break
    print("Thinking...\n")
    reviews = retriver.invoke(user_question)
    print(chain.invoke({"reviews": reviews, "question": user_question}))
