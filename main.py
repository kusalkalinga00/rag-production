from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

from langchain_openai import ChatOpenAI

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    # Test OpenAI
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("What is the capital of France?")
    print(f"OpenAI response: {response}")
    

if __name__ == "__main__":
    main()
