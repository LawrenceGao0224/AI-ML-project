import autogen

config_list = [
    {
        'model': "gpt-3.5-turbo",
        'api_key': "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"
    }
]

llm_config = {
    "request_timeout" : 600,
    "seed" : 42,
    "config_list" : config_list,
    "tempurture": 0,
}

assistant = autogen.AssistantAgent(
    name = "CTO",
    llm_config=llm_config,
    system_message="Chief technical officer of a global giant tech company"
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={'work_dir': 'web'},
    llm_config=llm_config,
    system_message="""Reply TERMINATE if the task have been solved at full satisfication.
    Otherwise, reply CONTINUE, or the reson why the task is not solved yet."""
)

task = """Give me a summary of the article: https://microsoft.github.io/autogen/0.2/blog/2024/03/03/AutoGen-Update/ """

user_proxy.initiate_chat(
    assistant,
    message=task
)

task2 = """
Change the code in the file you just created to instead output numbers 1 to 200
"""

user_proxy.initiate_chat(
    assistant,
    message=task2
)