# AWS Bedrock Agent Observability with W&B Weave

This guide focuses specifically on the integration between AWS Bedrock Agents and Weights & Biases (W&B) Weave for observability and tracing.

## How the Core Scripts Work

### create_bedrock_agent.py

This script handles the complete lifecycle of creating and preparing a Bedrock Agent:

1. **IAM Role Setup**: Creates or reuses an IAM role with the necessary permissions for Bedrock Agents
2. **Agent Creation**: Creates a new Bedrock Agent with specified foundation model and instructions
3. **Agent Preparation**: Initiates preparation and waits for the agent to reach the PREPARED state
4. **Alias Creation**: Creates an alias for the agent (required for invocation)
5. **Detail Storage**: Saves agent IDs and details for future tracing sessions

The script includes robust error handling and status checking to ensure the agent is ready for use.

### trace_bedrock_wandb.py

This script handles the tracing and observability aspects:

1. **W&B Configuration**: Sets up the connection to Weights & Biases Weave
2. **OpenTelemetry Setup**: Configures OTLP exporters to send traces to W&B
3. **Permissions Check**: Optionally verifies the user has permissions to invoke the agent
4. **Agent Invocation**: Invokes the Bedrock Agent with OTel instrumentation
5. **Response Processing**: Handles streaming or non-streaming responses
6. **Telemetry Export**: Ensures all traces are properly flushed to W&B Weave

The script includes interactive prompts for agent details and questions, making it easy to use.

## Code Organization: Essential vs Helper Functions

Both scripts have been organized to clearly separate essential core functionality from helper functions:

### Essential Functions

These functions form the minimal set required to perform the main tasks:

**In create_bedrock_agent.py:**
- `create_bedrock_clients()`: Initialize AWS clients
- `ensure_bedrock_agent_role()`: Create/validate IAM roles
- `create_bedrock_agent()`: Create the agent
- `prepare_agent()`: Start agent preparation
- `create_agent_alias()`: Create an agent alias

**In trace_bedrock_wandb.py:**
- `create_bedrock_clients()`: Initialize AWS clients
- `configure_weave_otel()`: Configure OpenTelemetry export
- `invoke_bedrock_agent()`: Invoke the agent
- `process_response()`: Process agent responses
- `trace_agent_invocation()`: Core tracing function

### Helper Functions

These provide robustness, error handling, and convenience:

**In create_bedrock_agent.py:**
- `get_agent_status()`: Check agent status with error handling
- `wait_for_agent_prepared()`: Wait with timeout and retries
- `alias_exists()`: Validate alias existence
- `save_agent_details()`: Persist agent IDs for future use
- `get_saved_agent_details()`: Retrieve saved agent IDs
- `create_or_reuse_agent()`: Intelligent agent selection logic

**In trace_bedrock_wandb.py:**
- `get_agent_model()`: Fetch agent model ID with error handling
- `process_streaming_response()`: Process streaming responses
- `check_agent_permissions()`: Validate permissions before invocation

This separation makes it easier to understand the essential requirements while benefiting from the more robust implementation.

## How the Integration Works

The integration uses OpenTelemetry to capture traces from AWS Bedrock Agent invocations and sends them to W&B Weave:

1. **Instrumentation Layer**: The `@instrument_agent_invocation` decorator wraps Bedrock Agent API calls
2. **Trace Collection**: Spans are created for different phases of agent execution
3. **OTLP Export**: Traces are sent to W&B using the OTLP HTTP exporter
4. **Weave UI**: Traces are visualized in the Weave UI for analysis

## Setting Up W&B Weave Integration

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. W&B Authentication

Add to your `.env` file:
```
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=your-entity/your-project
```

### 3. Configure OpenTelemetry Export

The integration automatically configures OpenTelemetry to send traces to W&B Weave:

```python
# From trace_bedrock_wandb.py
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
```

## Viewing Traces in W&B Weave

After running the tracing script, you can view your traces:

1. Navigate to: `https://wandb.ai/your-entity/your-project/traces`
2. Each trace corresponds to a single agent invocation
3. Click on a trace to view the detailed execution timeline
4. Explore span details by clicking on individual spans

### Key Trace Components

When viewing a trace in W&B Weave, you'll see:

- **Root Span**: The overall agent invocation
- **LLM Spans**: Foundation model (e.g., Claude) invocations
- **Action Group Spans**: Tool/action invocations
- **Knowledge Base Spans**: Retrievals from knowledge bases
- **Guardrail Spans**: Content safety and validation checks

### Understanding Span Attributes

Each span includes key attributes that help you understand the agent's behavior:

- **Input/Output**: Prompts and completions
- **Token Usage**: Tokens used by different operations
- **Timing**: Duration of each operation
- **Metadata**: Agent IDs, session information, user context
