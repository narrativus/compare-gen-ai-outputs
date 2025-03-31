# Compare Gen AI Outputs

Compare Gen AI Outputs is a project designed to experiment with and compare the outputs of different generative AI models. Initially, it integrates with OpenAI’s GPT-4o model and is built to be modular and easily extendable for future integrations (e.g., Gemini, Llama, Claude, DeepSeek).

## Project Structure

genai-tester/
├── .gitignore
├── README.md
├── .env                  # (For secrets, not committed)
├── pyproject.toml        # Managed by Poetry
├── poetry.lock           # Managed by Poetry
├── config/
│   └── models_config.yaml   # Holds catalog of available models, etc.
├── src/
│   ├── __init__.py
│   └── model_interface.py   # Code for interfacing with various models
├── notebooks/
│   └── gen_ai_models.ipynb    # Notebook to try out different Gen AI models
└── logs/
    └── reasoning.log      # File to store verbose chain-of-thought outputs


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