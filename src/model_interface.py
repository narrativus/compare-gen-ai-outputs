import os
import logging
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file.")

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

    Parameters:
        prompt (str): The input prompt for the model.
        model (str): The OpenAI model to use (default "gpt-4o").
        log_verbose (bool): Whether to log the verbose chain-of-thought output.

    Returns:
        str: The final output from the model.
    """
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        # Extract the final output using attribute access
        final_output = response.choices[0].message.content

        # Capture verbose chain-of-thought if available
        # (If 'verbose' is provided as an attribute, otherwise default to a message)
        verbose_info = getattr(
            response, "verbose", "No verbose chain-of-thought provided."
        )
        if log_verbose:
            logger.info(f"Prompt: {prompt}\nVerbose Output: {verbose_info}")

        return final_output
    except Exception as e:
        logger.error(f"Error querying OpenAI: {e}")
        return f"Error: {e}"


def compare_models(prompt, model1, model2, evaluation_model):
    """
    Query two models with the same prompt, then use a third model to evaluate their responses.

    Parameters:
        prompt (str): The original prompt.
        model1 (str): The identifier for the first model (e.g., "gpt-4o").
        model2 (str): The identifier for the second model (e.g., "gpt-3.5-turbo").
        evaluation_model (str): The model used to evaluate the two responses.

    Returns:
        tuple: (output1, output2, evaluation_output)
    """
    # Query the first and second models
    output1 = query_openai(prompt, model=model1)
    output2 = query_openai(prompt, model=model2)

    # Construct an evaluation prompt that includes the original prompt and both responses.
    evaluation_prompt = f"""
    Evaluate the following two responses to the given prompt. Provide an evaluation of which response is better and explain why.
    
    Prompt:
    {prompt}
    
    Response from {model1}:
    {output1}
    
    Response from {model2}:
    {output2}
    
    Your evaluation:
    """

    # Query the evaluation model with the evaluation prompt
    evaluation_output = query_openai(evaluation_prompt, model=evaluation_model)

    return output1, output2, evaluation_output
