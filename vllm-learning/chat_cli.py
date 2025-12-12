from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


def main():
    client = OpenAI(
        api_key = openai_api_key,
        base_url = openai_api_base
    )
    messages = [{"role": "system", "content": "you are a helpful assistant."}]

    while True:
        user_msg = input("Input: ")
        msg = {"role": "user", "content": user_msg}
        messages.append(msg)
        completion = client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct", messages=messages)
        print(f"assistant: {completion.choices[0].message.content}")
        messages.append(completion.choices[0].message)

if __name__ == "__main__":
    main()