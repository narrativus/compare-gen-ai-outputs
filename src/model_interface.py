# model_interface.py

import os
import logging
import requests  # Make sure requests is imported
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve keys for OpenAI (required)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file.")

# For Gemini and Mistral, we leave the warnings as placeholders.
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
if not GOOGLE_GEMINI_API_KEY:
    logging.warning(
        "GOOGLE_GEMINI_API_KEY is not set. Gemini queries will use simulated responses."
    )

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    logging.warning(
        "MISTRAL_API_KEY is not set. Mistral queries will use simulated responses."
    )

# Retrieve Hugging Face API key for Meta models (if available)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    logging.warning(
        "HUGGINGFACE_API_KEY is not set. Meta queries will use simulated responses."
    )

# Now import and instantiate the OpenAI client
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# Set up logging to capture chain-of-thought outputs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file_path = os.path.join(os.path.dirname(__file__), "..", "logs", "reasoning.log")
handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def query_openai(prompt, model="gpt-4o", log_verbose=True):
    """
    Query the OpenAI model with the given prompt.
    """
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        final_output = response.choices[0].message.content
        verbose_info = getattr(
            response, "verbose", "No verbose chain-of-thought provided."
        )
        if log_verbose:
            logger.info(
                f"OpenAI Query: Prompt: {prompt}\nVerbose Output: {verbose_info}"
            )
        return final_output
    except Exception as e:
        logger.error(f"Error querying OpenAI: {e}")
        return f"Error: {e}"


def query_gemini(prompt, model="gemini-default", log_verbose=True):
    """
    Query the Gemini model with the given prompt.

    This function currently simulates a response.
    """
    simulated_response = f"[Gemini simulated response for {model}] {prompt}"
    if log_verbose:
        logger.info(
            f"Gemini Query: Prompt: {prompt} | Model: {model} | Response: {simulated_response}"
        )
    return simulated_response


def query_mistral(prompt, model="mistral-default", log_verbose=True):
    """
    Query the Mistral model with the given prompt.

    This function currently simulates a response.
    """
    simulated_response = f"[Mistral simulated response for {model}] {prompt}"
    if log_verbose:
        logger.info(
            f"Mistral Query: Prompt: {prompt} | Model: {model} | Response: {simulated_response}"
        )
    return simulated_response


def query_meta(prompt, model="meta-default", log_verbose=True):
    """
    Query Meta's open source model via the Hugging Face Inference API.

    If HUGGINGFACE_API_KEY is provided, this function makes an HTTP request.
    Otherwise, it returns a simulated response.
    """
    if not HUGGINGFACE_API_KEY:
        simulated_response = f"[Meta simulated response for {model}] {prompt}"
        if log_verbose:
            logger.info(
                f"Meta Query (simulated): Prompt: {prompt} | Model: {model} | Response: {simulated_response}"
            )
        return simulated_response

    # Construct the endpoint URL. For example, if you want to use Meta's Llama-2 chat model hosted on Hugging Face,
    # you might use an endpoint like: "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat"
    # For now, we'll assume `model` holds the appropriate Hugging Face model identifier.
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()
        # Assume output is a list of responses with key "generated_text"
        if (
            isinstance(output, list)
            and len(output) > 0
            and "generated_text" in output[0]
        ):
            final_output = output[0]["generated_text"]
        else:
            final_output = str(output)
        if log_verbose:
            logger.info(
                f"Meta Query: Prompt: {prompt} | Model: {model} | Response: {final_output}"
            )
        return final_output
    except Exception as e:
        logger.error(f"Error querying Meta model via Hugging Face: {e}")
        return f"Error: {e}"


def query_model(prompt, provider="openai", model=None, log_verbose=True):
    """
    Generic query function to handle different model providers.
    """
    provider = provider.lower()
    if provider == "openai":
        if model is None:
            model = "gpt-4o"
        return query_openai(prompt, model=model, log_verbose=log_verbose)
    elif provider == "gemini":
        if model is None:
            model = "gemini-default"
        return query_gemini(prompt, model=model, log_verbose=log_verbose)
    elif provider == "mistral":
        if model is None:
            model = "mistral-default"
        return query_mistral(prompt, model=model, log_verbose=log_verbose)
    elif provider == "meta":
        if model is None:
            model = "meta-default"
        return query_meta(prompt, model=model, log_verbose=log_verbose)
    else:
        logger.error(f"Provider {provider} is not supported.")
        return f"Error: Provider {provider} is not supported."


def compare_models(
    prompt, model1, provider1, model2, provider2, evaluation_provider, evaluation_model
):
    """
    Query two models (possibly from different providers) with the same prompt,
    then use a third model to evaluate their responses.
    """
    output1 = query_model(prompt, provider=provider1, model=model1)
    output2 = query_model(prompt, provider=provider2, model=model2)
    evaluation_prompt = f"""
    Evaluate the following two responses to the given prompt. Provide an evaluation of which response is better and explain why.
    
    Prompt:
    {prompt}
    
    Response from {provider1} ({model1}):
    {output1}
    
    Response from {provider2} ({model2}):
    {output2}
    
    Your evaluation:
    """
    evaluation_output = query_model(
        evaluation_prompt, provider=evaluation_provider, model=evaluation_model
    )
    return output1, output2, evaluation_output
