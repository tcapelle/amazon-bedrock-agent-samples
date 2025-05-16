"""
Trace AWS Bedrock Agent invocations with OpenTelemetry and Weights & Biases Weave.

This script focuses solely on tracing functionality - no agent creation is performed.
"""

import os
import base64
import uuid
import boto3
from dotenv import load_dotenv
from rich.console import Console

# Import core instrumentation functionality
from core.timer_lib import timer
from core import instrument_agent_invocation, flush_telemetry
import weave

# Load environment variables from .env file
load_dotenv(".env")

console = Console()

#######################################################################
#                        ESSENTIAL FUNCTIONS                          #
# These functions represent the core functionality needed to trace    #
# agent invocations with OpenTelemetry and W&B Weave                  #
#######################################################################

def create_bedrock_clients():
    """Create and return Bedrock clients."""
    # Validate required environment variables
    required_vars = ["AWS_DEFAULT_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    for var in required_vars:
        if not os.environ.get(var):
            raise ValueError(f"Missing required environment variable: {var}")
    
    # Create the clients for Amazon Bedrock
    bedrock_client = boto3.client(
        service_name='bedrock-agent',
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),  # Optional
    )
    
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-agent-runtime",
        region_name=os.environ["AWS_DEFAULT_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),  # Optional
    )
    
    return bedrock_client, bedrock_runtime_client

def configure_weave_otel(wandb_api_key, project_name):
    """Configure OpenTelemetry to send traces to Weights & Biases Weave."""
    # W&B Weave endpoint for OTLP
    wandb_api_url = "https://trace.wandb.ai"
    
    # Create auth header (base64 encoded)
    auth = base64.b64encode(f"api:{wandb_api_key}".encode()).decode()
    
    # Set OpenTelemetry environment variables
    os.environ["OTEL_SERVICE_NAME"] = 'wandb'
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{wandb_api_url}/otel/v1/traces"
    
    # Set headers with authorization and project ID
    headers = f"Authorization=Basic {auth},project_id={project_name}"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers
    
    console.print(f"✅ OpenTelemetry configured to send traces to Weave project: {project_name}")
    return wandb_api_url

def invoke_bedrock_agent(inputText, agentId, agentAliasId, sessionId, **kwargs):
    """Invoke a Bedrock Agent."""
    _, bedrock_rt_client = create_bedrock_clients()
    
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

def process_response(response, streaming=False):
    """Process the response from Bedrock Agent."""
    if streaming:
        return process_streaming_response(response.get("completion", []))
    
    # Process non-streaming response
    completion = ""
    for event in response.get("completion", []):
        chunk = event.get("chunk", {})
        if "bytes" in chunk:
            completion += chunk["bytes"].decode()
    
    return completion

def trace_agent_invocation(agent_id, agent_alias_id, question, project_name, wandb_api_key, streaming=False):
    """Trace a Bedrock Agent invocation and send telemetry to W&B Weave."""
    # Create a unique session ID
    session_id = f"session-{uuid.uuid4().hex}"
    
    # User information and tags for filtering in Weave
    user_id = os.getenv("USER_ID", "user123")
    tags = ["bedrock-agent", "example", "development"]
    
    console.rule("Invoking Bedrock Agent with Instrumentation")
    console.print(f"Agent ID: {agent_id}")
    console.print(f"Agent Alias ID: {agent_alias_id}")
    console.print(f"Question: {question}")
    console.print(f"Streaming mode: {'Enabled' if streaming else 'Disabled'}")
    
    try:
        # Get the model ID for the agent
        agent_model_id = get_agent_model(agent_id)
        if not agent_model_id:
            agent_model_id = os.getenv("FOUNDATION_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
            console.print(f"Using default model ID: {agent_model_id}")
        else:
            console.print(f"Using agent model: {agent_model_id}")
        
        # Configure W&B Weave API URL
        wandb_api_url = configure_weave_otel(wandb_api_key, project_name)
        
        # Use the instrumentation decorator to wrap the invoke function
        instrumented_invoke = instrument_agent_invocation(invoke_bedrock_agent)
        
        # Streaming is prone to more issues, so prefer non-streaming for more reliable tracing
        if streaming:
            console.print("⚠️ Note: Streaming mode may generate more span-related warnings")
        
        try:
            # Invoke the agent with instrumentation
            response = instrumented_invoke(
                inputText=question,
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                show_traces=True,
                SAVE_TRACE_LOGS=True,
                userId=user_id,
                tags=tags,
                trace_id=str(uuid.uuid4()),
                project_name=project_name,
                wandb_api_key=wandb_api_key,
                wandb_api_url=wandb_api_url,
                streaming=streaming,
                model_id=agent_model_id,
            )
            
            # Process the response
            completion_text = process_response(response, streaming)
            console.print(f"Response: [green]{completion_text}[/green]")
            
        except Exception as invoke_error:
            if "accessDeniedException" in str(invoke_error):
                console.print("[bold red]❌ Access Denied Error:[/bold red]")
                console.print("The IAM role or user you're using doesn't have permission to invoke this agent.")
                console.print("\n[bold yellow]Possible solutions:[/bold yellow]")
                console.print("1. Verify the agent ID and alias ID are correct")
                console.print("2. Make sure your IAM role has 'bedrock:InvokeAgent' permissions")
                console.print("3. Check if you have access to this specific agent")
                console.print("4. Verify the agent is in PREPARED state")
            elif "ResourceNotFoundException" in str(invoke_error):
                console.print("[bold red]❌ Resource Not Found Error:[/bold red]")
                console.print("The agent or alias ID specified doesn't exist or isn't visible to your account.")
            else:
                console.print(f"❌ Error during agent invocation: {str(invoke_error)}")
            return None
        
        # Ensure all telemetry is sent
        flush_telemetry()
        console.print(f"✅ Traces sent to W&B Weave: {project_name}")
        console.print(f"View your traces at: https://wandb.ai/{project_name.split('/')[0]}/{project_name.split('/')[1]}/traces")
        
        return completion_text
    except Exception as e:
        console.print(f"❌ Error in tracing setup: {str(e)}")
        return None

#######################################################################
#                      HELPER FUNCTIONS                               #
# These functions provide error handling, response processing, and    #
# permission checking to make the tracing process more robust         #
#######################################################################

def get_agent_model(agent_id):
    """Get the foundation model used by an agent."""
    try:
        bedrock_client, _ = create_bedrock_clients()
        response = bedrock_client.get_agent(agentId=agent_id)
        return response["agent"]["foundationModel"]
    except Exception as e:
        console.print(f"❌ Error getting agent model: {str(e)}")
        return None

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
        console.print(f"\nError processing stream: {e}")
    return full_response

def check_agent_permissions(agent_id, agent_alias_id):
    """Test if the user has permission to access and invoke the agent."""
    try:
        bedrock_client, _ = create_bedrock_clients()
        
        # Try to get the agent details
        console.print(f"Checking access to agent {agent_id}...")
        agent_info = bedrock_client.get_agent(agentId=agent_id)
        console.print(f"✅ Agent {agent_id} exists and is accessible")
        
        # Check agent status
        if "agent" in agent_info and "agentStatus" in agent_info["agent"]:
            status = agent_info["agent"]["agentStatus"]
            console.print(f"Agent status: {status}")
            
            if status != "PREPARED":
                console.print(f"⚠️ Warning: Agent is in {status} state, not PREPARED")
                console.print("The agent must be in PREPARED state to be invoked")
                return False
        
        # If alias ID is provided, check that too
        if agent_alias_id:
            console.print(f"Checking access to alias {agent_alias_id}...")
            try:
                aliases = bedrock_client.list_agent_aliases(agentId=agent_id)
                alias_exists = any(alias["agentAliasId"] == agent_alias_id 
                                  for alias in aliases.get("agentAliasSummaries", []))
                
                if alias_exists:
                    console.print(f"✅ Alias {agent_alias_id} exists and is accessible")
                else:
                    console.print(f"❌ Alias {agent_alias_id} not found for agent {agent_id}")
                    return False
            except Exception as alias_error:
                console.print(f"❌ Error checking alias: {str(alias_error)}")
                return False
        
        console.print("✅ Permissions check passed! You should be able to invoke this agent.")
        return True
        
    except Exception as e:
        if "AccessDeniedException" in str(e) or "accessDeniedException" in str(e):
            console.print(f"❌ Access denied while checking agent {agent_id}")
            console.print("You do not have permission to access this agent")
        elif "ResourceNotFoundException" in str(e):
            console.print(f"❌ Agent {agent_id} not found")
        else:
            console.print(f"❌ Error checking agent permissions: {str(e)}")
        return False

def main():
    """Simple script to trace an existing Bedrock Agent."""
    console.rule("AWS Bedrock Agent Tracing with W&B Weave OpenTelemetry")
    
    # Get W&B API Key
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        wandb_api_key = console.input("Enter your W&B API key: ")
    
    # Configure project name
    project_name = os.getenv("WANDB_PROJECT", "wandb/otel_test") 
    
    # Initialize Weave
    weave.init(project_name=project_name)
    
    # Get agent ID and alias ID
    agent_id = os.getenv("AGENT_ID")
    if not agent_id:
        agent_id = console.input("Enter your Bedrock Agent ID: ")
    
    agent_alias_id = os.getenv("AGENT_ALIAS_ID")
    if not agent_alias_id:
        agent_alias_id = console.input("Enter your Bedrock Agent Alias ID: ")
    
    # Optionally test permissions before attempting to invoke
    should_check = os.getenv("CHECK_PERMISSIONS", "").lower() == "true"
    if not should_check and not os.getenv("CHECK_PERMISSIONS"):
        should_check = console.input("Would you like to test permissions before invoking? (y/n): ").lower().startswith("y")
    
    if should_check:
        console.rule("Checking Agent Permissions")
        if not check_agent_permissions(agent_id, agent_alias_id):
            try_anyway = console.input("Permission check failed. Try anyway? (y/n): ").lower().startswith("y")
            if not try_anyway:
                console.print("Exiting without invocation.")
                return
    
    # Get question from environment variable, argument, or prompt user
    question = os.getenv("QUESTION")
    if not question:
        question = console.input("Enter your question for the agent: ")
    
    # Streaming mode
    streaming_input = os.getenv("STREAMING", "").lower()
    if streaming_input in ("true", "false"):
        streaming = streaming_input == "true"
    else:
        streaming = console.input("Enable streaming? (y/n): ").lower().startswith("y")
    
    # Trace the agent invocation
    trace_agent_invocation(
        agent_id=agent_id,
        agent_alias_id=agent_alias_id,
        question=question,
        project_name=project_name,
        wandb_api_key=wandb_api_key,
        streaming=streaming
    )

if __name__ == "__main__":
    main()