import autogen

def main():
    config_list = autogen.config_list_from_json(
        env_or_file = "CONFIG_LIST.json"
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

    user_proxy.initiate_chat(assistant, message="Plot a chart of Tesla and Meta stock price change.")

if __name__ == "__main__":
    main()
