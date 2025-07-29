from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

def get_conversational_answer(top_texts, query, groq_api):
    template = (
        "You are an expert conversational AI assistant. "
        "Always respond to the user's question in a friendly, natural, and helpful manner, using the information provided in the context below. "
        "If the context does not contain the answer, politely say so instead of guessing.\n\n"
        "Context:\n{context}\n\n"
        "User Question: {question}\n\n"
        "Your Answer (be clear, simple, gentle, engaging, short and accurate, referencing context where possible, and not too long. "
        "It should be super natural and conversational): and don't answer to the question out of context and always answer in 2 or 3 sentence, it should be very short but not too short and must be conversational"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    formatted_prompt = prompt.format(context="\n\n".join(top_texts), question=query)
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=512,
        api_key=groq_api,
    )
    response = llm.invoke(formatted_prompt)
    return response.content