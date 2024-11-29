if __name__ == "__main__":
    import streamlit as st
    from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains.question_answering import load_qa_chain

    OPENAI_KEY=""

    #Upload PDF files
    st.header("My First Chatbot")

    with st.sidebar:
        st.title("Your Documents")
        file = st.file_uploader("upload a file and start asking questions ", type="pdf")
    
    #extract text here
    if file is not None:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
          
    #break it into chunks 
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function = len
        )

        chunks = text_splitter.split_text(text)
        #st.write(chunks)

        #generating embeddings
        embeddings = OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)
        
        #creating vector store - faISS
        vector_store = FAISS.from_texts(chunks,embeddings)

        #getting user query
        user_question = st.text_input("type your questions here")

        if user_question:
            match = vector_store.similarity_search(user_question)
            #st.write(match)

        #define the LLM
        llm = ChatOpenAI(
            openai_key = OPENAI_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )

        #output results
        #chain -> take the question , get relevant document , pass it into llm , generate the output
        chain = load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)

        st.write(response)        
