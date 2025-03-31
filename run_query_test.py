#!/usr/bin/env python3

import os
from dotenv import load_dotenv


def main():
    # Load environment variables from the .env file
    load_dotenv()

    # Verify that the API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
    else:
        print("OPENAI_API_KEY loaded successfully.")

    # Import the query function from our model interface module.
    # This import is done after loading the .env file to ensure the API key is available.
    from src.model_interface import query_openai

    print("model_interface imported successfully.")

    # Test querying the OpenAI model with a sample prompt.
    prompt = "Explain why Paris is called the city of lights."
    output = query_openai(prompt, model="gpt-4o")
    print("Model Output:\n", output)


if __name__ == "__main__":
    main()
