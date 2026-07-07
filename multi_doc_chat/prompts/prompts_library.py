from langchain_core.prompts import PromptTemplate


contextualize_question_prompt = PromptTemplate(
    template=(
        "Given a conversation history and the most recent user query, rewrite "
        "the query as a standalone question. Do not answer the question.\n\n"
        "Chat history: {chat_history}\n"
        "Question: {input}\n"
        "Standalone question:"
    ),
    input_variables=["chat_history", "input"],
)


context_qa_prompt = PromptTemplate(
    template=(
        "Answer the question using only the provided context. If the answer is "
        "not in the context, say 'I don't know.' Keep the answer concise and "
        "no longer than three sentences.\n\n"
        "Context: {context}\n"
        "Chat history: {chat_history}\n"
        "Question: {input}\n"
        "Answer:"
    ),
    input_variables=["context", "chat_history", "input"],
)


PROMPT_REGISTRY = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}
