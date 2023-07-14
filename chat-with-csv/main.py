import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import pandas as pd


def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your Csv 🍺")
    st.header("Ask your Csv 🍺")

    user_csv = st.file_uploader("Upload your csv file", type="csv")

    if user_csv is not None:
        data = pd.read_csv(user_csv, encoding="gbk")
        data.to_csv("data.csv", index=False)
        st.success("File saved successfully!")

        agent = create_csv_agent(
            OpenAI(temperature=0),
            "data.csv",
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        # 循环 输入框
        flag = True
        i = 0
        while flag:
            user_question = st.text_input("Ask a question about your csv:", key=i)
            flag = False
            if user_question is not None and user_question != "":
                response = agent.run(user_question)
                st.write(response)
                flag = True
                i += 1


if __name__ == '__main__':
    main()
