import autogen

def main():
    config_list=autogen.config_list_from_dotenv(
        ".env",
        {"gpt-3.5-turbo": "OPENAI_API_KEY"}
    )

    assistant = autogen.AssistantAgent(
        name = "Assistant",
        llm_config={
            "config_list" : config_list
        }
    )

    user_proxy = autogen.UserProxyAgent(
        name = "User",
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False
        }
    )

    user_proxy.initiate_chat(assistant, message="Give me a summary of the article: https://microsoft.github.io/autogen/0.2/blog/2024/03/03/AutoGen-Update/")

if __name__ == "__main__":
    main()
