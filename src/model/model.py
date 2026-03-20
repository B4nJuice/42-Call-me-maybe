import llm_sdk
import asyncio


class LLMModel:
    def __init__(self):
        self.model: llm_sdk.Small_LLM_Model =\
            llm_sdk.Small_LLM_Model(device="cpu")

    async def get_prompt_response(self, prompt: str, context) -> str:
        prompt_executor: PromptExecutor = PromptExecutor(self.model)
        asyncio.create_task(
            asyncio.to_thread(
                prompt_executor.get_prompt_response,
                prompt,
                context
            )
        )
        return prompt_executor


class PromptExecutor:
    def __init__(self, model):
        self.token: int = 0
        self.model: llm_sdk.Small_LLM_Model = model
        self.prompt_response: str = ""

    def get_prompt_response(self, prompt: str, context) -> str:
        token_prompt: list[int] = self.model.encode(
                f"{context}Input= {prompt} \nFunction="
            ).tolist()[0]
        colon_token_id: list[int] = self.model.encode(":").tolist()[0]
        result: str = ""
        parameters_given: bool = False

        while True:
            logits = self.model.get_logits_from_input_ids(token_prompt)
            for cid in colon_token_id:
                logits[cid] = float("-inf")
            next_id = logits.index(max(logits))
            token_prompt.append(next_id)
            next_text: str = self.model.decode(next_id)
            if next_text.count("\n"):
                #prendre dans les definition pour savoir quelle fonction utiliser et mettre les parametres
                if parameters_given:
                    token_prompt.pop(-1)
                    to_add: str = next_text.replace("\n", "") + "|END|"
                    token_prompt += self.model.encode(to_add).tolist()[0]
                    result += to_add
                else:
                    token_prompt += self.model.encode("Parameters=").tolist()[0]
                    result += "\nParameters="
                    parameters_given = True
            else:
                result += next_text

            self.token += 1
            # print(next_text, end='', flush=True)
            print(result)
            if result.strip().endswith("|END|"):
                break

        self.prompt_response = f"Prompt= {prompt}\nFunction=" + result
        return self.prompt_response
