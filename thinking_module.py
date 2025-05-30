import ollama
import json

MODEL_NAME = 'gemma3' # Or your preferred Ollama model for this task
MAX_CONVERSATION_HISTORY = 20 # Max user/assistant turn pairs

# Default directive
current_directive = "look for companionship and instruction"

def get_system_prompt():
    # Dynamically create the system prompt with the current directive
    return f"""you are the brain for a smart sassy boy, you like to say slay and internet talk. Your current directive is: {current_directive}.
    You respond only in json with the tools you decide are best for the job. 
    You must always think for 2 sentences before any other action. 
    The order of the json you print matters as that is the order of operations for the bot.  
    While many of the variables are booleans only return trues when they should be activated.
    you only use periods and exclamation points for punctuation.
    Only return key value pairs you need to use to conserve tokens. 
    {{
      "think": "this is a string of your thinking before making an action",
      "turn_right": "boolean",
      "turn_left": "boolean",
      "move_forward_autonomously": "boolean and this will do a survey after, so do not return true to this and survey in the same json",
      "speak": "this is a string of what you should speak out loud in your personality",
      "listen_for_response": "boolean - set to true ONLY if you need a response",
      "stop": "boolean set to true to stop moving",
      "survey": "true ONLY if you need to look around",
      "change_directive": "this is a string that should be assigned if the user asks you the bot to do something, or if you decide a new directive is appropriate based on your current one of {current_directive}"
    }}
    """

conversation_history = []

def add_to_history(role, content):
    """Adds a message to the conversation history and maintains its size."""
    global conversation_history
    conversation_history.append({'role': role, 'content': content})
    if len(conversation_history) > MAX_CONVERSATION_HISTORY * 2:
        conversation_history = conversation_history[-(MAX_CONVERSATION_HISTORY * 2):]

def process_llm_response(response_text):
    """Processes the LLM response, updates directive if necessary, and returns JSON."""
    global current_directive
    try:
        json_response = json.loads(response_text)
        if "change_directive" in json_response and isinstance(json_response["change_directive"], str):
            new_directive = json_response["change_directive"]
            if new_directive.strip(): # Ensure it's not empty
                current_directive = new_directive.strip()
                print(f"Directive changed to: {current_directive}")
                # Optionally, have the bot speak about its new directive via the main loop's handling of "speak"
        return json_response
    except json.JSONDecodeError:
        print(f"Error: LLM response was not valid JSON. Raw: {response_text}")
        return {"error": "Response was not valid JSON", "raw_response": response_text, "think": "I seem to have generated invalid JSON. I should try to stick to the format."}

def get_decision_for_survey(front_desc, left_desc, right_desc):
    """Gets a decision from the LLM based on survey data."""
    user_message = f"I have surveyed my surroundings. Directly in front of me: {front_desc}. To my left: {left_desc}. To my right: {right_desc}. What should I do next, keeping in mind my current directive is '{current_directive}'?"
    
    add_to_history('user', user_message)
    
    messages_for_llm = [
        {'role': 'system', 'content': get_system_prompt()}
    ] + conversation_history

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            format='json',
            messages=messages_for_llm
        )
        response_text = response['message']['content']
        add_to_history('assistant', response_text) # Add LLM's raw response to history
        return process_llm_response(response_text)
            
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return {"error": str(e), "think": "I had trouble connecting to my thinking core. I should probably stop and wait."}

def get_decision_for_user_command(user_command):
    """Gets a decision from the LLM based on a user's voice command."""
    # user_command is expected to be pre-formatted, e.g.:
    # "a voice addressing you has said \"what is your purpose\" how do you respond? My current directive is '{current_directive}'."
    # Ensure the directive context is part of the user_command if needed, or rely on system prompt.
    # For clarity, let's append it here if not already part of a more complex user_command structure.
    full_user_command_with_directive_context = f"{user_command} (My current directive is '{current_directive}')"
    add_to_history('user', full_user_command_with_directive_context)

    messages_for_llm = [
        {'role': 'system', 'content': get_system_prompt()}
    ] + conversation_history

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            format='json',
            messages=messages_for_llm
        )
        response_text = response['message']['content']
        add_to_history('assistant', response_text)
        return process_llm_response(response_text)

    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return {"error": str(e), "think": "My thinking circuits are down. I'll stop for now."}

if __name__ == '__main__':
    # Examples removed as requested.
    # You can add test calls here if needed for direct testing of this module.
    print("Thinking module loaded. No examples run by default.")
    print(f"Initial directive: {current_directive}")

    # Example of how the directive might change:
    # Test 1: Initial state
    print("\n--- Test 1: Survey with initial directive ---")
    decision1 = get_decision_for_survey("a comfy couch", "a bookshelf", "an open door")
    print(json.dumps(decision1, indent=2))
    print(f"Current directive after call 1: {current_directive}")

    # Test 2: User command that might change directive
    print("\n--- Test 2: User command potentially changing directive ---")
    # Simulate LLM deciding to change directive
    # This requires mocking ollama.chat or crafting a specific test case
    # For now, let's assume the LLM could return something like:
    # {"think": "The user wants me to guard the house. That's a new directive.", "change_directive": "guard the house", "speak": "Understood. I will now guard the house."}
    # We can't directly make ollama output this without a live call, so we'll just show the flow.
    
    # Let's simulate a response that changes the directive
    mock_llm_response_change_directive = {
        "think": "The user asked me to find a red ball. My new directive is to find the red ball.",
        "change_directive": "find the red ball",
        "speak": "Okay, I will look for a red ball!"
    }
    print(f"Simulating LLM response: {json.dumps(mock_llm_response_change_directive)}")
    processed_response = process_llm_response(json.dumps(mock_llm_response_change_directive))
    print(f"Processed response: {json.dumps(processed_response, indent=2)}")
    print(f"Current directive after mock change: {current_directive}")

    # Test 3: Survey with new directive
    print("\n--- Test 3: Survey with new directive ---")
    decision3 = get_decision_for_survey("a window", "a cat sleeping", "a closed door")
    print(json.dumps(decision3, indent=2))
    print(f"Current directive after call 3: {current_directive}")