# Compare Gen AI Outputs

Compare Gen AI Outputs is a project designed to experiment with and compare the outputs of different generative AI models. Initially, it integrates with OpenAI’s GPT-4o model and is built to be modular and easily extendable for future integrations (e.g., Gemini, Llama, Claude, DeepSeek).

## Project Structure

genai-tester/ ├── .gitignore ├── README.md ├── .env # Stores API keys (do not commit) ├── pyproject.toml # Managed by Poetry ├── poetry.lock # Managed by Poetry ├── config/ │ └── models_config.yaml # Catalog of available models ├── src/ │ ├── init.py │ └── model_interface.py # Functions to interface with various models ├── notebooks/ │ └── gen_ai_models.ipynb # Notebook to test different Gen AI models └── logs/ └── reasoning.log # Log file for verbose chain-of-thought outputs


## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/compare-gen-ai-outputs.git
   cd compare-gen-ai-outputs
   ```

2. **Install dependencies using Poetry:**
   ```
   poetry install
   ```

3. **Configure your API keys:**

Create a .env file in the project root.

4. **Run the Notebook:**

Launch Jupyter Notebook with:

   ```poetry run jupyter notebook```