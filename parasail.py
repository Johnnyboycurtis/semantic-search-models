import os
import requests
import json
import time
from typing import Dict, Any, Optional


class ParasailClient:
    """
    A client for interacting with the Parasail Chat Completion API, utilizing
    a persistent requests.Session for efficiency and connection pooling.
    """

    BASE_URL = "https://api.parasail.io/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, model_name: str = "allenai/Olmo-3-7B-Instruct", retries: Optional[int] = None, system_instructions: Optional[str] = None):
        """
        Initializes the client, loads the API key, and sets up the session.
        """
        self.api_key = api_key or os.getenv("PARASAIL_API_KEY")
        self.MAX_RETRIES = 1 or None
        self.RETRY_DELAY_SECONDS = 0.5
        self.SYSTEM_INSTRUCTIONS = system_instructions
        self.model_name = model_name

        if not self.api_key:
            raise ValueError(
                "API Key not found. Please provide it during initialization "
                "or set the 'PARASAIL_API_KEY' environment variable."
            )

        # Initialize the session object
        self.session = requests.Session()

        # Configure session headers globally for all requests made through this client
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def __enter__(self):
        """Allows the client to be used as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures the session is closed cleanly upon exiting the 'with' block."""
        self.session.close()

    def generate_completion(
        self, prompt: str, model: str = "google/gemma-3-27b-it"
    ) -> Dict[str, Any]:
        """
        Sends a POST request to generate a chat completion using the persistent session.

        Args:
            prompt: The user's input text.
            model: The model identifier to use.

        Returns:
            The JSON response data from the API.
        """
        messages = []
        if self.SYSTEM_INSTRUCTIONS:
            messages = [{"role": "system", "content": self.SYSTEM_INSTRUCTIONS}]
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model_name, "messages": messages}

        # --- RETRY LOGIC START ---
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                print(f"Attempt {attempt}/{self.MAX_RETRIES}...")

                response = self.session.post(self.BASE_URL, json=payload)

                # This line raises an HTTPError for 4xx or 5xx status codes
                response.raise_for_status()

                # Success: Return the JSON data immediately
                return response.json()

            except requests.exceptions.HTTPError as e:
                # Handle HTTP errors (4xx or 5xx)
                status_code = response.status_code

                # Retry only on transient server errors (500-599)
                if 500 <= status_code < 600 and attempt < self.MAX_RETRIES:
                    print(
                        f"Transient Server Error ({status_code}). Waiting {self.RETRY_DELAY_SECONDS}s before retry."
                    )
                    time.sleep(self.RETRY_DELAY_SECONDS * attempt)
                    continue  # Continue to the next loop iteration

                # Fail fast on client errors (4xx) or if retries are exhausted
                print(f"Fatal HTTP Error ({status_code}) encountered.")
                raise e

            except requests.exceptions.RequestException as e:
                # Handle general network issues (DNS failure, connection timeout, etc.)
                if attempt < self.MAX_RETRIES:
                    print(
                        f"Connection Error: {type(e).__name__}. Waiting {self.RETRY_DELAY_SECONDS}s before retry."
                    )
                    time.sleep(self.RETRY_DELAY_SECONDS)
                    continue

                # Final failure due to network issue
                print("Exhausted network retries.")
                raise e
        # --- RETRY LOGIC END ---

        # This line should technically only be hit if MAX_RETRIES was 0,
        # but serves as a final catch-all error return.
        return {"error": "Exhausted all configured retry attempts."}

    def chat(self, message: str):
        api_response = self.generate_completion(prompt=message)
        return api_response["choices"][0]["message"]["content"]


# --- Example Usage ---

# 1. Set up the Environment Variable (Necessary for the example to run)
# For testing purposes, we simulate setting it here:
os.environ["PARASAIL_API_KEY"] = "YOUR_SECRET_KEY_HERE_12345"

def demo(api_key):

    user_prompt = "What is the capital of New York?"
    model_name = "allenai/Olmo-3-7B-Instruct"

    instructions = """# SYSTEM INSTRUCTIONS: QUERY GENERATOR
    Your task is to generate 1-3 search queries given a body of text. 
    The search queries should be contextually relevant to the body of text.
    The search queries should be what a human would generate (i.e. simple search messages)
    *   Only generate queries when there is significant contextual and semantic knowledge.
    *   If context in the body of text is minimal, you can return an empty list `[]`.

    ## OUTPUT FORMAT
    ```json
    [
        "query one based on text A",
        "query two focused on detail B",
        "a final, broader query C"
    ]
    ```
    """
    blob = """### TEXT:
The domestic rabbit (Oryctolagus cuniculus domesticus) is the domesticated form of the European rabbit. There are hundreds of rabbit breeds originating from all over the world. Rabbits were first domesticated and used for food and fur by the Romans. Rabbits may be housed inside, but the idea of the domestic rabbit as a house companion, a so-called house rabbit (similar to a house cat), was only strongly promoted starting with publications in the 1980s. Rabbits can be trained to use a litter box and taught to come when called, but require exercise and can damage a house or injure themselves if it has not been suitably prepared, based on their innate need to chew. Accidental interactions between pet rabbits and wild rabbits, while seemingly harmless, have been strongly discouraged due to the species' different temperaments as well as wild rabbits potentially carrying diseases.

Unwanted pet rabbits sometimes end up in animal shelters, especially after the Easter season. In 2017, they were the United States' third most abandoned pet. Some of them go on to be adopted and become family pets in various forms. Because their wild counterparts have become invasive in Australia, pet rabbits are banned in the state of Queensland. Domestic rabbits bred for generations under human supervision to be docile will be less able to care or fend for themselves, if they are abandoned or escape from captivity.
"""
    blob = """### TEXT:
    ## Project Goals

*   [ ] Release initial versions of a general-purpose sentence embedding model.
*   [ ] Release an accompanying cross-encoder reranking model.
*   [ ] Publish the training scripts and configurations for all released models.
*   [ ] Add performance benchmarks on common information retrieval tasks.

## License

Distributed under the MIT License.
"""

    user_prompt = blob

    print(f"Query: '{user_prompt}' using model: {model_name}\n")

    # Initialize the client and use it as a context manager (recommended)
    try:
        with ParasailClient(api_key=api_key, system_instructions=instructions, model_name=model_name) as client:
            # Call the function
            api_response = client.generate_completion(
                prompt=user_prompt, model=model_name
            )

            # Display results (Mocked output since the actual API is not available)
            if "error" not in api_response:
                print("\n--- API Success ---")
                # Assuming the API returns standard completion structure:
                try:
                    content = api_response["choices"][0]["message"]["content"]
                    print(f"Response Content: {content}")
                except (TypeError, KeyError):
                    print("Raw Response (Success, but unexpected structure):")
                    print(json.dumps(api_response, indent=2))
            else:
                print("\n--- API Failure ---")
                print(f"Error Details: {api_response['error']}")

    except ValueError as e:
        print(f"Initialization Failed: {e}")
        exit(1)
    # The session is automatically closed when exiting the 'with' block
