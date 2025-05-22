import requests
import json
# import os # Potentially for API keys from env vars later

# --- Helper Functions for Each Service ---

def _summarize_ollama(text_to_summarize: str, ollama_model: str, ollama_api_url: str) -> dict:
    """
    Summarizes text using a local Ollama API.
    """
    if not ollama_api_url:
        return {'summary': None, 'error': "Ollama API URL not provided."}
    if not ollama_model:
        return {'summary': None, 'error': "Ollama model not specified."}

    # Ensure the URL has a scheme
    if not ollama_api_url.startswith(('http://', 'https://')):
        ollama_api_url = 'http://' + ollama_api_url

    # Prefer /api/chat for more structured interaction, similar to OpenAI
    # Simple prompt for summarization
    prompt = f"Summarize the following text in a concise manner:\n\n{text_to_summarize}\n\nSummary:"
    payload = {
        "model": ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False # For a single response
    }
    
    # Try /api/chat first
    chat_api_url = f"{ollama_api_url.rstrip('/')}/api/chat"
    generate_api_url = f"{ollama_api_url.rstrip('/')}/api/generate" # Fallback

    try:
        response = requests.post(chat_api_url, json=payload, timeout=60) # Increased timeout
        response.raise_for_status()
        response_data = response.json()
        if 'message' in response_data and 'content' in response_data['message']:
            return {'summary': response_data['message']['content'].strip(), 'error': None}
        # Fallback for older Ollama versions or if /api/chat response is unexpected
        elif 'response' in response_data: # This is typical for /api/generate
             return {'summary': response_data['response'].strip(), 'error': None}
        else:
            return {'summary': None, 'error': f"Unexpected response structure from Ollama /api/chat: {response_data}"}

    except requests.exceptions.RequestException as e_chat:
        # If /api/chat fails, try /api/generate as a fallback
        # print(f"Ollama /api/chat failed: {e_chat}. Trying /api/generate...")
        payload_generate = {
            "model": ollama_model,
            "prompt": prompt, # /api/generate uses "prompt"
            "stream": False
        }
        try:
            response = requests.post(generate_api_url, json=payload_generate, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            if 'response' in response_data:
                return {'summary': response_data['response'].strip(), 'error': None}
            else:
                 return {'summary': None, 'error': f"Unexpected response structure from Ollama /api/generate: {response_data}"}
        except requests.exceptions.RequestException as e_generate:
            return {'summary': None, 'error': f"Ollama API request failed for both /api/chat and /api/generate. Chat error: {e_chat}. Generate error: {e_generate}"}
    except json.JSONDecodeError:
        return {'summary': None, 'error': "Failed to decode Ollama API response as JSON."}


def _summarize_gemini(text_to_summarize: str, api_key: str) -> dict:
    """
    Summarizes text using the Google Gemini API.
    """
    if not api_key:
        return {'summary': None, 'error': "Gemini API key not provided."}

    # Using gemini-1.5-flash as it's typically faster and cheaper for summarization
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": f"Summarize the following text concisely:\n\n{text_to_summarize}"}]
        }],
        "generationConfig": { # Optional: control output
            "temperature": 0.7,
            "maxOutputTokens": 250,
        }
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        response_data = response.json()

        if (response_data.get('candidates') and
                response_data['candidates'][0].get('content') and
                response_data['candidates'][0]['content'].get('parts') and
                response_data['candidates'][0]['content']['parts'][0].get('text')):
            summary = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
            return {'summary': summary, 'error': None}
        else:
            error_detail = response_data.get('error', {}).get('message', 'Unknown structure')
            if 'promptFeedback' in response_data: # Check for safety blocks
                 error_detail = f"Content blocked by API. Feedback: {response_data['promptFeedback']}"
            return {'summary': None, 'error': f"Failed to extract summary from Gemini response. Detail: {error_detail}. Response: {response_data}"}

    except requests.exceptions.RequestException as e:
        return {'summary': None, 'error': f"Gemini API request failed: {e}"}
    except json.JSONDecodeError:
        return {'summary': None, 'error': "Failed to decode Gemini API response as JSON."}


def _summarize_openrouter(text_to_summarize: str, api_key: str, model: str = "mistralai/mistral-7b-instruct-v0.2") -> dict:
    """
    Summarizes text using the OpenRouter API.
    """
    if not api_key:
        return {'summary': None, 'error': "OpenRouter API key not provided."}

    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost/news-aggregator" # Placeholder, as per OpenRouter docs
    }
    payload = {
        "model": model, # Default to a known good model, can be parameterized further
        "messages": [
            {"role": "user", "content": f"Please summarize the following text concisely:\n\n{text_to_summarize}"}
        ],
        "temperature": 0.7,
        "max_tokens": 250
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        response_data = response.json()

        if (response_data.get('choices') and
                response_data['choices'][0].get('message') and
                response_data['choices'][0]['message'].get('content')):
            summary = response_data['choices'][0]['message']['content'].strip()
            return {'summary': summary, 'error': None}
        else:
            return {'summary': None, 'error': f"Failed to extract summary from OpenRouter response. Response: {response_data}"}

    except requests.exceptions.RequestException as e:
        error_message = f"OpenRouter API request failed: {e}"
        if response and response.text:
            error_message += f" - Response: {response.text}"
        return {'summary': None, 'error': error_message}
    except json.JSONDecodeError:
        return {'summary': None, 'error': "Failed to decode OpenRouter API response as JSON."}


# --- Main Dispatch Function ---

def summarize_text(text_to_summarize: str, service: str, api_key: str = None, 
                   ollama_model: str = None, ollama_api_url: str = None, 
                   openrouter_model: str = "mistralai/mistral-7b-instruct-v0.2") -> dict:
    """
    Summarizes text using the specified service.
    """
    if not text_to_summarize or not text_to_summarize.strip():
        return {'summary': None, 'error': "Input text is empty or whitespace."}

    if service == "ollama":
        return _summarize_ollama(text_to_summarize, ollama_model, ollama_api_url)
    elif service == "gemini":
        return _summarize_gemini(text_to_summarize, api_key)
    elif service == "openrouter":
        return _summarize_openrouter(text_to_summarize, api_key, model=openrouter_model)
    else:
        return {'summary': None, 'error': f"Unsupported summarization service: {service}"}


# --- Test Block ---
if __name__ == '__main__':
    sample_text = (
        "Artificial intelligence (AI) is rapidly transforming various industries, from healthcare to finance. "
        "Machine learning algorithms are becoming increasingly sophisticated, enabling predictive analytics and automation. "
        "However, ethical considerations surrounding AI, such as bias in algorithms and job displacement, need careful attention. "
        "Researchers are working on developing more transparent and fair AI systems. The future of AI promises significant "
        "advancements but also requires responsible development and deployment to ensure its benefits are shared widely."
        "This text is intentionally made a bit longer to provide enough content for summarization algorithms to process effectively, "
        "leading to a more meaningful and representative summary of the input."
    )
    print(f"Original Text ({len(sample_text.split())} words):\n{sample_text}\n")

    # Test Ollama (assuming Ollama is running and 'llama3' model is available)
    # Set your Ollama URL if it's not the default.
    # Ensure you have pulled the model: `ollama pull llama3`
    print("--- Testing Ollama ---")
    ollama_api_url_test = "http://localhost:11434" # Default, change if yours is different
    ollama_model_test = "llama3" # Change if you use a different model
    
    # Check if Ollama server is reachable (basic check)
    ollama_reachable = False
    try:
        requests.get(f"{ollama_api_url_test}/api/tags", timeout=5) # /api/tags is a lightweight endpoint
        ollama_reachable = True
        print(f"Ollama server detected at {ollama_api_url_test}. Attempting summarization...")
    except requests.exceptions.ConnectionError:
        print(f"Ollama server not reachable at {ollama_api_url_test}. Skipping Ollama test.")
    
    if ollama_reachable:
        ollama_summary_result = summarize_text(
            sample_text, 
            "ollama", 
            ollama_model=ollama_model_test, 
            ollama_api_url=ollama_api_url_test
        )
        if ollama_summary_result['error']:
            print(f"Ollama Error: {ollama_summary_result['error']}")
        else:
            print(f"Ollama Summary ({len(ollama_summary_result['summary'].split())} words): {ollama_summary_result['summary']}")
    print("-" * 25)

    # Test Gemini (requires a valid API key)
    # IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual key to test.
    # You can also set it as an environment variable: os.environ.get("GEMINI_API_KEY")
    print("\n--- Testing Google Gemini ---")
    gemini_api_key = "YOUR_GEMINI_API_KEY" # Replace with your key
    if gemini_api_key == "YOUR_GEMINI_API_KEY" or not gemini_api_key:
        print("Gemini API key not set. Skipping Gemini test. (Replace 'YOUR_GEMINI_API_KEY' in the script to test)")
    else:
        print("Attempting Gemini summarization...")
        gemini_summary_result = summarize_text(sample_text, "gemini", api_key=gemini_api_key)
        if gemini_summary_result['error']:
            print(f"Gemini Error: {gemini_summary_result['error']}")
        else:
            print(f"Gemini Summary ({len(gemini_summary_result['summary'].split())} words): {gemini_summary_result['summary']}")
    print("-" * 25)

    # Test OpenRouter (requires a valid API key)
    # IMPORTANT: Replace "YOUR_OPENROUTER_API_KEY" with your actual key to test.
    # You can also set it as an environment variable: os.environ.get("OPENROUTER_API_KEY")
    print("\n--- Testing OpenRouter ---")
    openrouter_api_key = "YOUR_OPENROUTER_API_KEY" # Replace with your key
    # Example of using a different model, e.g., a smaller, faster one if available and suitable
    # openrouter_model_test = "google/gemma-7b-it" 
    openrouter_model_test = "mistralai/mistral-7b-instruct-v0.2" # Default from function

    if openrouter_api_key == "YOUR_OPENROUTER_API_KEY" or not openrouter_api_key:
        print("OpenRouter API key not set. Skipping OpenRouter test. (Replace 'YOUR_OPENROUTER_API_KEY' in the script to test)")
    else:
        print(f"Attempting OpenRouter summarization with model {openrouter_model_test}...")
        openrouter_summary_result = summarize_text(
            sample_text, 
            "openrouter", 
            api_key=openrouter_api_key,
            openrouter_model=openrouter_model_test 
        )
        if openrouter_summary_result['error']:
            print(f"OpenRouter Error: {openrouter_summary_result['error']}")
        else:
            print(f"OpenRouter Summary ({len(openrouter_summary_result['summary'].split())} words): {openrouter_summary_result['summary']}")
    print("-" * 25)

    print("\nNote: For API services (Gemini, OpenRouter), ensure you have replaced placeholder API keys with your actual keys.")
    print("For Ollama, ensure the server is running, the specified model is downloaded, and the API URL is correct.")
    print("Summarization quality and speed will vary based on the model and service used.")
