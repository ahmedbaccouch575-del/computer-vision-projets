import streamlit as st
import os
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool
#################################
### initialize agent ##########
#################################

tools = [ImageCaptionTool(),ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
memory_key="chat_history",
   k=5,
  return_messages=True,
)
llm = ChatGoogleGenerativeAI(
    google_api_key="AQ.Ab8RN6JHKj5hLOIbDf1jiT20pTCQDXnVGYlGbqRxl81bcq6OTQ",
    temperature=0,
    model="gemini-3.8-flash",  # use a current model name shown in AI Studio
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method="generate",
)

# set title
st.title('Ask a question to an image')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])
if file:
    # display image
    st.image(file, use_container_width=True)

    # text input
    user_question = st.text_input('Ask a question about your image:')

    #################################
    ### compute agent response ### 
    from tempfile import NamedTemporaryFile       
    with st.spinner(text="In progress..."):
      with NamedTemporaryFile(dir='.', delete=False) as f:
        f.write(file.getbuffer())
        f.flush()
        image_path = f.name

        response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
    # write agent response
    if user_question and user_question != "":
      st.write(response)