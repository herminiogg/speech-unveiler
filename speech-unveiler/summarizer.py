from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import http.client
import json
import torch

class LLMSummarizer:
    def summary_to_file(self, summary, file):
        with open(file, "w", encoding="utf-8") as f:
            f.write(summary)

    def retrieve_text_from_file(self, file):
        with open(file, "r") as f:
            return f.read()

    def generate_prompt(self, text, max_words, language):
        return f"Given the following text, summarize it in {language} and max {max_words} words. Output just a plain summary and do not include any additional content, markdown or lists. \n\"{text}\""

class BuiltinLLMSummarizer(LLMSummarizer):
    def __init__(self):
        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )

    def summarize(self, file, max_words=200, language="english"):
        text = self.retrieve_text_from_file(file)
        if text:
            prompt = self.generate_prompt(text, max_words, language)
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
            )
            model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_words * 10 # This is not a direct correlation but it should help to avoid infinite generations.
                )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            _ = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return content
        else:
            raise Exception(f"File {file} has no contents")


class LocalLLMSummarizerAPI(LLMSummarizer):
    def __init__(self, endpoint, path, llm_model=None):
        self.endpoint = endpoint
        self.path = path
        self.llm_model = llm_model

    def summarize(self, file, max_words=200, language="english"):
        text = self.retrieve_text_from_file(file)
        if text:
            conn = http.client.HTTPConnection(self.endpoint)
            headers = {'Content-type': 'application/json'}
            prompt = self.generate_prompt(text, max_words, language)
            messages = [
                {"role": "user", "content": prompt}
            ]
            json_data = json.dumps({
                "messages": messages
            }) if not self.llm_model else json.dumps({
                "model": self.llm_model,
                "messages": messages
            })
            conn.request('POST', self.path, json_data, headers)
            return json.loads(conn.getresponse().read().decode())['choices'][0]['message']['content']

        else:
            raise Exception(f"File {file} has no contents")