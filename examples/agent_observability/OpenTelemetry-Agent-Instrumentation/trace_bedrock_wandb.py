"""
You should have an Agent ID and it should be `prepared`
"""



import os
import boto3
import uuid
from rich.console import Console

console = Console()

# Create the client to invoke Agents in Amazon Bedrock:
br_agents_runtime = boto3.client("bedrock-agent-runtime")
bedrock_client = boto3.client('bedrock-agent')
session_id = uuid.uuid4().hex

console.rule("Starting Bedrock Agent invocation...")
agent_id = "4GSXYOD7GM"  # <- Configure your Bedrock Agent ID
agent_alias_id = "TSTALIASID"  # <- Optionally set a different Alias ID if you have one


console.print(f"Trying to invoke alias {agent_alias_id} of agent {agent_id}...")
agent_resp = br_agents_runtime.invoke_agent(
    agentAliasId=agent_alias_id,
    agentId=agent_id,
    inputText="Hello!",
    sessionId=session_id,
)
if "completion" in agent_resp:
    console.print("✅ Got response")
else:
    raise ValueError(f"No 'completion' in agent response:\n{agent_resp}")

console.print(agent_resp)

completion = ""

for event in agent_resp.get("completion"):
    chunk = event["chunk"]
    completion += chunk["bytes"].decode()

console.print(f"Completion: [green]{completion}[/green]")


###################################################################################
console.rule("Telemetry data sent to WandB")

wandb_api_url = "https://trace.wandb.ai"
if wandb_api_key := os.getenv("WANDB_API_KEY"):
    console.print("WANDB_API_KEY found in environment variables")
else:
    wandb_api_key = console.input("Enter your WandB API key: ")

import os
import base64
import time
import boto3
import uuid
import json
from core.timer_lib import timer
from core import instrument_agent_invocation, flush_telemetry
import weave

@weave.op
@instrument_agent_invocation
def invoke_bedrock_agent(
    inputText: str, agentId: str, agentAliasId: str, sessionId: str, **kwargs
):
    """Invoke a Bedrock Agent with instrumentation for Langfuse."""
    # Create Bedrock client
    bedrock_rt_client = boto3.client("bedrock-agent-runtime")
    use_streaming = kwargs.get("streaming", False)
    invoke_params = {
        "inputText": inputText,
        "agentId": agentId,
        "agentAliasId": agentAliasId,
        "sessionId": sessionId,
        "enableTrace": True,  # Required for instrumentation
    }

    # Add streaming configurations if needed
    if use_streaming:
        invoke_params["streamingConfigurations"] = {
            "applyGuardrailInterval": 10,
            "streamFinalResponse": True,
        }
    response = bedrock_rt_client.invoke_agent(**invoke_params)
    return response

@weave.op
def process_streaming_response(stream):
    """Process a streaming response from Bedrock Agent."""
    full_response = ""
    try:
        for event in stream:
            # Convert event to dictionary if it's a botocore Event object
            event_dict = (
                event.to_response_dict()
                if hasattr(event, "to_response_dict")
                else event
            )
            if "chunk" in event_dict:
                chunk_data = event_dict["chunk"]
                if "bytes" in chunk_data:
                    output_bytes = chunk_data["bytes"]
                    # Convert bytes to string if needed
                    if isinstance(output_bytes, bytes):
                        output_text = output_bytes.decode("utf-8")
                    else:
                        output_text = str(output_bytes)
                    full_response += output_text
    except Exception as e:
        print(f"\nError processing stream: {e}")
    return full_response

##########################################
console.rule("Configuration OTEL W&B")


os.environ["OTEL_SERVICE_NAME"] = 'wandb'

# Create auth header
AUTH = base64.b64encode(f"api:{wandb_api_key}".encode()).decode()

## Payload
project_name = "wandb/otel_test"

# Set OpenTelemetry environment variables for Langfuse
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{wandb_api_url}/otel/v1/traces"
# Combine Authorization and project_id headers
headers = f"Authorization=Basic {AUTH},project_id={project_name}"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers

# Agent configuration
agentId = agent_id
agentAliasId = agent_alias_id
sessionId = f"session-{int(time.time())}"

# User information
userId = "capecape"  

console.rule("Get Agent info")
agent_info = bedrock_client.get_agent(agentId=agent_id)
console.print(agent_info)
agent_model_id = agent_info["agent"]["foundationModel"]

# we should probably retrieve the model from the agent itself
# agent_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Tags for filtering in Langfuse
tags = ["bedrock-agent", "example", "development"]

# Generate a custom trace ID
trace_id = str(uuid.uuid4())

# Your prompt and streaming mode
question = "Tell me a long story about cats" # your prompt to the agent
streaming = False

console.rule("Starting Bedrock Agent invocation...")
weave.init(project_name=project_name)
# Single invocation that works for both streaming and non-streaming
response = invoke_bedrock_agent(
    inputText=question,
    agentId=agentId,
    agentAliasId=agentAliasId,
    sessionId=sessionId,
    show_traces=True,
    SAVE_TRACE_LOGS=True,
    userId=userId,
    tags=tags,
    trace_id=trace_id,
    project_name=project_name,
    wandb_api_key=wandb_api_key,
    wandb_api_url=wandb_api_url,
    streaming=streaming,
    model_id=agent_model_id,
)