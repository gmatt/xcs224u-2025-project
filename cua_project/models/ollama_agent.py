from ollama import chat

from mm_agents.agent import PromptAgent


class OllamaAgent(PromptAgent):
    def call_llm(self, payload):
        messages = payload["messages"]
        for message in messages:
            assert len(message["content"]) in [1, 2]
            content = None
            if message["content"][0]["type"] == "text":
                content = message["content"][0]["text"]
            if (
                len(message["content"]) > 1
                and message["content"][1]["type"] == "image_url"
            ):
                message["images"] = [
                    message["content"][1]["image_url"]["url"].replace(
                        "data:image/png;base64,", ""
                    )
                ]
            message["content"] = content

        # Remove all but last image to save context.
        for message in messages[:-1]:
            if "images" in message:
                del message["images"]

        response = chat(
            model="gemma3:4b",
            messages=messages,
        )
        return response.message.content
