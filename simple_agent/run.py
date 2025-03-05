import os
from langchain_community.llms.openai import OpenAI
from langchain_core.tools.simple import Tool
from langchain.agents.initialize import initialize_agent
from dotenv import load_dotenv
from typing import Dict
from naptha_sdk.schemas import *
from naptha_sdk.user import sign_consumer_id
from naptha_sdk.utils import get_logger
from simple_agent.schemas import InputSchema

logger = get_logger(__name__)
load_dotenv()
def add_numbers_tool(query: str) -> str:
    try:
        numbers = [int(num) for num in query.split() if num.isdigit()]
        return str(sum(numbers))
    except ValueError:
        return "Error: Could not parse integers from query."


def simple_agent():
    tools = [
        Tool(
            name="Addition Tool",
            func=add_numbers_tool,
            description="Useful for adding numbers. Provide text with integers."
        )
    ]

    # Create and return the agent executor
    return initialize_agent(
        tools,
        OpenAI(temperature=0),
        agent="zero-shot-react-description",
        verbose=True
    )

def run(module_run: Dict, *args, **kwargs):
    """
    Modified run function that creates and executes the agent.
    If 'func_name' is 'simple_agent', we build the agent and run it
    with the 'description' provided in func_input_data.
    """

    # Parse the input schema
    module_run = AgentRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)

    # Check which function we want to call
    func_to_call = globals().get(module_run.inputs.func_name)
    if not func_to_call:
        raise ValueError(f"Function '{module_run.inputs.func_name}' not found.")

    # If func_name requests 'agent_name', create and run the agent
    if module_run.inputs.func_name == "simple_agent":
        user_question = module_run.inputs.func_input_data.get("description", "")
        expected_output = module_run.inputs.func_input_data.get("expected_output", "Analysis results")
        if not user_question:
            return {"error": "No question provided in func_input_data['description']."}
        
        # Create the agent executor
        agent_executor = simple_agent()
        
        # Execute the agent with the user's question and return the result
        result = agent_executor.run(user_question)
        
        # Return the result directly (not as a dictionary)
        return result
    else:
        # Fallback for other functions
        import inspect
        sig = inspect.signature(func_to_call)
        if len(sig.parameters) == 0:
            return func_to_call()
        else:
            tool_input_class = (
                globals().get(module_run.inputs.input_type)
                if module_run.inputs.input_type else None
            )
            input_data = (
                tool_input_class(**module_run.inputs.func_input_data)
                if tool_input_class else module_run.inputs.func_input_data
            )
            return func_to_call(input_data)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()
    deployment = asyncio.run(
        setup_module_deployment(
            "agent",
            "simple_agent/configs/deployment.json",
            node_url=os.getenv("NODE_URL"),
            user_id=None,
            load_persona_data=False,
            is_subdeployment=False
        )
    )

    example_inputs = {
        "description": "What is the sum of 243435343 and 3254354353",
        "expected_output": "the sum "
    }

    input_params = {
        "func_name": "simple_agent",
        "func_input_data": example_inputs,
    }

    module_run = {
        "inputs": input_params,
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, os.getenv("PRIVATE_KEY"))
    }

    response = run(module_run)
    print(response)