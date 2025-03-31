import os
import logging
import openai
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Set up logging to capture chain-of-thought outputs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'logs', 'reasoning.log'))
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file.")

openai.api_key = OPENAI_API_KEY

def query_openai(prompt, model="gpt-4o", log_verbose=True):
    """
    Query the OpenAI model with the given prompt.

    Parameters:
        prompt (str): The input prompt for the model.
        model (str): The OpenAI model to use (default "gpt-4o").
        log_verbose (bool): If True, log verbose chain-of-thought outputs.

    Returns:
        str: The final output from the model.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract the final output
        final_output = response.choices[0].message.get('content', '')
        
        # Capture verbose chain-of-thought if available
        verbose_info = response.get("verbose", "No verbose chain-of-thought provided.")
        if log_verbose:
            logger.info(f"Prompt: {prompt}\nVerbose Output: {verbose_info}")
        
        return final_output
    except Exception as e:
        logger.error(f"Error querying OpenAI: {e}")
        return f"Error: {e}"
