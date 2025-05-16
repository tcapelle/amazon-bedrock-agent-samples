"""
Create an AWS Bedrock Agent with the necessary IAM role.

This script handles:
1. Creating an IAM role with minimum required permissions for Bedrock Agents
2. Creating a Bedrock Agent
3. Preparing the agent
4. Creating an alias for the agent
"""

import os
import time
import uuid 
import json
import boto3
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables from .env file
load_dotenv(".env")

console = Console()

#######################################################################
#                        ESSENTIAL FUNCTIONS                          #
# These functions represent the core functionality needed to create   #
# and prepare a Bedrock Agent                                         #
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

def ensure_bedrock_agent_role(role_name="bedrock-agent-execution-role"):
    """
    Ensures that an IAM role exists for Bedrock Agents with the minimum necessary permissions.
    If the role doesn't exist, it will create it.
    
    Args:
        role_name (str): The name of the IAM role to check or create
        
    Returns:
        str: The ARN of the IAM role
    """
    try:
        # Create an IAM client
        iam_client = boto3.client('iam')
        
        # Try to get the role to see if it exists
        try:
            response = iam_client.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            console.print(f"✅ Found existing IAM role: {role_name}")
            console.print(f"Role ARN: {role_arn}")
            return role_arn
        except iam_client.exceptions.NoSuchEntityException:
            console.print(f"IAM role {role_name} not found. Creating...")
        
        # Define the trust policy to allow Bedrock to assume this role
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Create the role
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for AWS Bedrock Agents to invoke models and access resources"
        )
        
        role_arn = response['Role']['Arn']
        
        # Define the minimum permissions policy for Bedrock Agents
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        # Create the policy
        policy_name = f"{role_name}-policy"
        policy_response = iam_client.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_document),
            Description="Minimum permissions for Bedrock Agents"
        )
        
        policy_arn = policy_response['Policy']['Arn']
        
        # Attach the policy to the role
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn=policy_arn
        )
        
        console.print(f"✅ Created new IAM role: {role_name}")
        console.print(f"Role ARN: {role_arn}")
        
        # IAM role propagation can take time
        console.print("Waiting 10 seconds for IAM role propagation...")
        time.sleep(10)
        
        return role_arn
    except Exception as e:
        console.print(f"❌ Error creating IAM role: {str(e)}")
        raise

def create_bedrock_agent(name, description, foundation_model, instruction, role_arn):
    """
    Create an AWS Bedrock Agent programmatically.
    
    Args:
        name (str): The name of the agent
        description (str): Description of the agent's purpose
        foundation_model (str): Model to use (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
        instruction (str): Instructions for the agent
        role_arn (str): IAM role ARN with required permissions
        
    Returns:
        dict: The response containing agent details including the agent ID
    """
    bedrock_client, _ = create_bedrock_clients()
    
    # Role ARN is required for agent creation
    if not role_arn:
        raise ValueError("IAM role ARN is required to create a Bedrock Agent")
    
    create_params = {
        "agentName": name,
        "agentResourceRoleArn": role_arn,
        "description": description,
        "foundationModel": foundation_model,
        "instruction": instruction,
        "idleSessionTTLInSeconds": 1800,  # 30 minutes
    }
    
    response = bedrock_client.create_agent(**create_params)
    
    console.print(f"✅ Created agent: {name}")
    console.print(f"Agent ID: {response['agent']['agentId']}")
    
    return response

def prepare_agent(agent_id):
    """
    Prepare an agent for use.
    
    Args:
        agent_id (str): The ID of the agent to prepare
        
    Returns:
        dict: The response containing the preparation status
    """
    bedrock_client, _ = create_bedrock_clients()
    
    response = bedrock_client.prepare_agent(agentId=agent_id)
    
    console.print(f"✅ Agent {agent_id} preparation started")
    return response

def create_agent_alias(agent_id, alias_name, description, max_retries=3):
    """
    Create an alias for a Bedrock Agent.
    
    Args:
        agent_id (str): The ID of the existing agent
        alias_name (str): Name for the alias
        description (str): Description of the alias
        max_retries (int): Maximum number of retries for creating the alias
        
    Returns:
        dict: The response containing alias details or None if it fails
    """
    bedrock_client, _ = create_bedrock_clients()
    
    # First, check agent status to ensure it's prepared
    status, error = get_agent_status(agent_id)
    if error or status != "PREPARED":
        console.print(f"❌ Cannot create alias: Agent is not in PREPARED state (current: {status})")
        console.print("Wait for the agent to be fully prepared before creating an alias")
        # Throwing an exception since this is a critical failure
        raise ValueError(f"Agent must be in PREPARED state to create an alias, current state: {status}")
    
    for attempt in range(max_retries):
        try:
            console.print(f"Creating alias '{alias_name}' for agent {agent_id} (attempt {attempt+1}/{max_retries})")
            response = bedrock_client.create_agent_alias(
                agentId=agent_id,
                agentAliasName=alias_name,
                description=description
            )
            
            console.print(f"✅ Created agent alias: {alias_name}")
            console.print(f"Agent Alias ID: {response['agentAlias']['agentAliasId']}")
            
            return response
        except Exception as e:
            console.print(f"⚠️ Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # Exponential backoff
                console.print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                console.print(f"❌ Failed to create agent alias after {max_retries} attempts")
                raise

#######################################################################
#                      HELPER FUNCTIONS                               #
# These functions provide error handling, status checking, and reuse  #
# capabilities to make the agent creation process more robust         #
#######################################################################

def get_agent_status(agent_id):
    """
    Safely get the status of an agent, handling possible errors.
    
    Args:
        agent_id (str): The ID of the agent to check
        
    Returns:
        tuple: (status, error_message) where status is the agent's status if available,
               or None if there was an error. error_message contains details of any error.
    """
    bedrock_client, _ = create_bedrock_clients()
    
    try:
        response = bedrock_client.get_agent(agentId=agent_id)
        
        # Check if agent field exists
        if "agent" not in response:
            return None, f"'agent' field not found in response: {response}"
            
        # The field is called 'agentStatus' and is inside the 'agent' object
        if "agentStatus" not in response["agent"]:
            return None, f"'agentStatus' field not found in agent response: {response['agent']}"
            
        return response["agent"]["agentStatus"], None
        
    except Exception as e:
        return None, str(e)

def wait_for_agent_prepared(agent_id, max_wait_seconds=300, check_interval=10):
    """
    Wait for an agent to be in the PREPARED state.
    
    Args:
        agent_id (str): The ID of the agent to check
        max_wait_seconds (int): Maximum wait time in seconds
        check_interval (int): Time between status checks in seconds
        
    Returns:
        bool: True if agent is prepared, False if timeout reached
    """
    start_time = time.time()
    
    console.print(f"Waiting for agent {agent_id} to be prepared...")
    
    while time.time() - start_time < max_wait_seconds:
        status, error = get_agent_status(agent_id)
        
        if error:
            # Print a more concise error message to avoid overwhelming output
            if len(error) > 200:
                error_summary = error[:200] + "... [truncated]"
                console.print(f"⚠️ Error checking agent status: {error_summary}")
                console.print("Will try again after a delay")
            else:
                console.print(f"⚠️ Error checking agent status: {error}")
                console.print("Will try again after a delay")
            time.sleep(check_interval)
            continue
            
        console.print(f"Agent status: {status}")
        
        # The exact status string should be "PREPARED"
        if status == "PREPARED":
            console.print(f"✅ Agent {agent_id} is now PREPARED")
            return True
        
        # If failed, stop waiting
        if status == "FAILED":
            console.print(f"❌ Agent {agent_id} preparation FAILED")
            return False
            
        console.print(f"Waiting {check_interval} seconds...")
        time.sleep(check_interval)
    
    console.print(f"⚠️ Timeout waiting for agent {agent_id} to be prepared")
    return False

def alias_exists(agent_id, alias_id):
    """
    Check if an alias exists for a given agent.
    
    Args:
        agent_id (str): The agent ID
        alias_id (str): The alias ID to check
        
    Returns:
        bool: True if the alias exists, False otherwise
    """
    try:
        bedrock_client, _ = create_bedrock_clients()
        aliases = bedrock_client.list_agent_aliases(agentId=agent_id)
        
        if "agentAliasSummaries" not in aliases:
            console.print(f"⚠️ No alias summaries found for agent {agent_id}")
            return False
            
        return any(alias["agentAliasId"] == alias_id for alias in aliases.get("agentAliasSummaries", []))
    except Exception as e:
        console.print(f"❌ Error checking alias existence: {str(e)}")
        return False

def save_agent_details(agent_id, agent_alias_id, file_path=".agent_details.json"):
    """
    Save agent and alias IDs to a file for future use.
    
    Args:
        agent_id (str): The agent ID
        agent_alias_id (str): The agent alias ID
        file_path (str): Path to save the details
    """
    details = {
        "agent_id": agent_id,
        "agent_alias_id": agent_alias_id,
        "created_at": time.time()
    }
    
    with open(file_path, "w") as f:
        json.dump(details, f, indent=2)
    
    console.print(f"✅ Saved agent details to {file_path}")

def get_saved_agent_details(file_path=".agent_details.json"):
    """
    Get saved agent details if available.
    
    Args:
        file_path (str): Path to read the details from
        
    Returns:
        dict or None: The agent details or None if not found
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                details = json.load(f)
            return details
        return None
    except Exception:
        return None

def create_or_reuse_agent():
    """
    Create a new Bedrock Agent or reuse an existing one.
    
    Returns:
        tuple: (agent_id, agent_alias_id, agent_model_id) or None if failed
    """
    console.rule("AWS Bedrock Agent Setup")
    
    # First check if we have an existing agent in environment variables
    agent_id = os.getenv("AGENT_ID")
    agent_alias_id = os.getenv("AGENT_ALIAS_ID")
    
    # If we have both, verify they exist and are valid
    if agent_id and agent_alias_id:
        try:
            bedrock_client, _ = create_bedrock_clients()
            agent_info = bedrock_client.get_agent(agentId=agent_id)
            agent_model_id = agent_info["agent"]["foundationModel"]
            
            # Check if the alias exists
            aliases = bedrock_client.list_agent_aliases(agentId=agent_id)
            alias_exists = any(alias["agentAliasId"] == agent_alias_id for alias in aliases.get("agentAliasSummaries", []))
            
            if alias_exists:
                console.print(f"✅ Using existing agent from environment variables")
                console.print(f"Agent ID: {agent_id}")
                console.print(f"Agent Alias ID: {agent_alias_id}")
                console.print(f"Agent Model: {agent_model_id}")
                return agent_id, agent_alias_id, agent_model_id
            else:
                console.print(f"⚠️ Agent {agent_id} exists but alias {agent_alias_id} not found")
        except Exception as e:
            console.print(f"⚠️ Error verifying agent from environment variables: {str(e)}")
    
    # Next check if we have saved agent details
    saved_details = get_saved_agent_details()
    if saved_details:
        try:
            agent_id = saved_details["agent_id"]
            agent_alias_id = saved_details["agent_alias_id"]
            
            bedrock_client, _ = create_bedrock_clients()
            agent_info = bedrock_client.get_agent(agentId=agent_id)
            agent_model_id = agent_info["agent"]["foundationModel"]
            
            console.print(f"✅ Using previously saved agent")
            console.print(f"Agent ID: {agent_id}")
            console.print(f"Agent Alias ID: {agent_alias_id}")
            console.print(f"Agent Model: {agent_model_id}")
            return agent_id, agent_alias_id, agent_model_id
        except Exception as e:
            console.print(f"⚠️ Error verifying saved agent: {str(e)}")
            console.print("Will create a new agent")
    
    # Create a new agent since we don't have a valid existing one
    
    # First, ensure we have a role
    agent_role_arn = os.getenv("AGENT_ROLE_ARN")
    if not agent_role_arn:
        console.print("⚠️ No AGENT_ROLE_ARN found in environment variables.")
        console.print("Attempting to create or reuse an IAM role for Bedrock agents...")
        try:
            agent_role_arn = ensure_bedrock_agent_role()
            console.print(f"Using role ARN: {agent_role_arn}")
        except Exception as e:
            console.print(f"❌ Failed to create IAM role: {str(e)}")
            console.print("Cannot create agent without a role ARN")
            return None
    
    try:
        # Create a new agent
        console.rule("Creating a new Bedrock Agent")
        foundation_model = os.getenv("FOUNDATION_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
        
        agent_response = create_bedrock_agent(
            name=f"TestAgent_{uuid.uuid4().hex[:8]}",
            description="A test agent for OpenTelemetry tracing",
            foundation_model=foundation_model,
            instruction="You are a helpful, friendly AI assistant that answers questions concisely.",
            role_arn=agent_role_arn
        )
        
        # Get the agent ID
        agent_id = agent_response["agent"]["agentId"]
        agent_model_id = agent_response["agent"]["foundationModel"]
        
        # Prepare the agent for use
        console.print("Preparing agent for use...")
        prepare_agent(agent_id)
        
        # Wait for the agent to be prepared before creating an alias
        if wait_for_agent_prepared(agent_id):
            # Create an alias
            try:
                alias_response = create_agent_alias(
                    agent_id=agent_id,
                    alias_name=f"test_alias_{uuid.uuid4().hex[:8]}",
                    description="Test alias for OpenTelemetry demo"
                )
                agent_alias_id = alias_response["agentAlias"]["agentAliasId"]
                
                # Save the details for future use
                save_agent_details(agent_id, agent_alias_id)
                
                return agent_id, agent_alias_id, agent_model_id
            except Exception as e:
                console.print(f"❌ Error creating agent alias: {str(e)}")
                console.print("Agent was created and prepared but alias creation failed.")
                console.print(f"You can manually create an alias for agent ID: {agent_id}")
                return None
        else:
            console.print("❌ Agent preparation failed or timed out.")
            console.print(f"The agent (ID: {agent_id}) might still be usable once preparation completes.")
            return None
            
    except Exception as e:
        console.print(f"❌ Error creating agent: {str(e)}")
        return None

if __name__ == "__main__":
    # When run directly, just create/reuse an agent and print the details
    agent_details = create_or_reuse_agent()
    
    if agent_details:
        agent_id, agent_alias_id, agent_model_id = agent_details
        console.print("\n===== Agent Details =====")
        console.print(f"Agent ID: {agent_id}")
        console.print(f"Agent Alias ID: {agent_alias_id}")
        console.print(f"Model ID: {agent_model_id}")
        console.print("Add these to your .env file or pass to trace_bedrock_wandb.py")
    else:
        console.print("❌ Failed to create or reuse an agent") 