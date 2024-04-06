import os

from openai import OpenAI
import openai
import json

from dotenv import load_dotenv

load_dotenv()



openai.api_key = os.environ['OPENAI_API_KEY']


client = OpenAI()

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def get_biggest_cities():
    """Get a list of the biggest cities in the world"""
    cities = ["San Francisco", "Tokyo", "Paris"]
    return json.dumps(cities)


messages = [
    {"role": "system", "content": "you give very short response and you should return answer  as a sentence "},
    {"role": "user", "content": "What's the weather like in biggest_cities"}
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_biggest_cities",
            "description": "Get a list of the biggest cities in the world. This function does not require any parameters.",
            "parameters": {}  # No parameters required for this function
        }
    }
]


def run_conversation():
    while True:
        print(messages)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        messages.append(response_message)  # extend conversation with assistant's reply

        if (not tool_calls) and response.choices[0].finish_reason == "stop":
            return messages[-1]

        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "get_current_weather": get_current_weather,
                "get_biggest_cities": get_biggest_cities,
            }  # only one function in this example, but you can have multiple
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                if function_name == "get_biggest_cities":
                    # No arguments needed for this function
                    function_response = function_to_call()
                else:
                    # Handle functions with arguments as before
                    function_args = json.loads(tool_call.function.arguments)
                    function_response = function_to_call(
                        location=function_args.get("location", ""),
                        unit=function_args.get("unit", "fahrenheit"))


                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response



print(run_conversation())
