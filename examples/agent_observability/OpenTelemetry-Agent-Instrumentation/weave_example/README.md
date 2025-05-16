# AWS Bedrock Agent Observability with W&B Weave

## Getting Started

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Setup environment variables in `.env` file:
   ```
   USER_ID=wandb_username
   WANDB_API_KEY=your_wandb_api_key
   AWS_DEFAULT_REGION=your_aws_region
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_SESSION_TOKEN=your_aws_session_token  # Optional
   ```

3. Create a Bedrock agent:
   ```bash
   python create_bedrock_agent.py
   ```

4. Trace agent invocations:
   ```bash
   python trace_bedrock_wandb.py
   ```
   Follow the interactive prompts or provide agent IDs from step 3.

## Script Overview

### create_bedrock_agent.py

Creates and prepares a Bedrock Agent with the necessary setup:
- Creates/reuses an IAM role with Bedrock permissions
- Creates a new Bedrock Agent with specified foundation model
- Prepares the agent and waits for PREPARED state
- Creates an alias for agent invocation
- Stores agent details for future tracing sessions

### trace_bedrock_wandb.py

Handles the tracing and observability with W&B Weave:
- Configures W&B and OpenTelemetry connection
- Sets up OTLP exporters to send traces to W&B
- Provides instrumentation for agent invocations
- Processes streaming or non-streaming responses
- Exports telemetry to W&B Weave for visualization

## Integration Details

The integration uses OpenTelemetry to capture and send Bedrock Agent traces to W&B Weave through the `core/` module.

### Core Module Structure

The `core/` directory contains the essential components:

- **agent.py**: Defines the `instrument_agent_invocation` decorator that wraps Bedrock API calls and manages the span hierarchy
- **configuration.py**: Configures the OpenTelemetry tracer provider and exporters
- **tracing.py**: Provides utilities for span management and telemetry flushing
- **constants.py**: Defines constants for span attributes and kinds
- **handlers.py**: Contains specialized handlers for different trace types (guardrails, preprocessing, failures)
- **processes.py**: Processes specific trace types (orchestration, post-processing)
- **streaming_wrapper.py**: Handles streaming responses with proper trace context

### Instrumentation Flow

1. **Initialization**: The `instrument_agent_invocation` decorator from `core/agent.py` wraps the agent invocation:
   ```python
   @instrument_agent_invocation
   def invoke_bedrock_agent(inputText, agentId, agentAliasId, sessionId, **kwargs):
       # Original invocation logic
   ```

2. **Span Hierarchy**: The instrumentation creates a parent span for the overall operation and child spans for each component:
   ```python
   with tracer.start_as_current_span(name=f"Bedrock Agent: {agentId}", kind=SpanKind.CLIENT) as root_span:
       # Agent invocation happens here
       response = func(inputText=inputText, ...)
   ```

3. **Trace Processing**: The `SpanManager` class tracks and manages spans throughout the execution:
   ```python
   class SpanManager:
       def __init__(self):
           # Track spans by component type and trace ID
           self.spans = {}
           self.active_traces = {...}
           self.special_spans = {...}
   ```

4. **Response Analysis**: For each trace event in the response, handlers create corresponding spans:
   ```python
   def process_trace_event(trace_data, parent_span):
       # Determine trace type and route to appropriate handler
       if "guardrailTrace" in trace:
           handle_guardrail_pre(trace_data, parent_span)
       elif "orchestrationTrace" in trace:
           process_orchestration_trace(trace_data, parent_span)
   ```

### OTEL Configuration

The integration configures OpenTelemetry with the `create_tracer_provider()` function in `configuration.py`:

```python
def create_tracer_provider(service_name, environment, resource_attributes, endpoint, headers):
    # Setup tracer provider with proper resource attributes
    tracer_provider = TracerProvider(resource=Resource.create(attributes))
    
    # Configure OTLP exporter for W&B Weave
    otlp_exporter = OTLPSpanExporter(endpoint=wandb_api_url, headers=headers)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set as global tracer
    trace.set_tracer_provider(tracer_provider)
```

The W&B Weave-specific setup happens in `trace_bedrock_wandb.py`:

```python
def configure_weave_otel(wandb_api_key, project_name):
    wandb_api_url = "https://trace.wandb.ai"
    auth = base64.b64encode(f"api:{wandb_api_key}".encode()).decode()
    
    os.environ["OTEL_SERVICE_NAME"] = 'wandb'
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{wandb_api_url}/otel/v1/traces"
    headers = f"Authorization=Basic {auth},project_id={project_name}"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers
```

### Viewing and Customization

#### Viewing Traces in W&B

Traces in W&B Weave show:
- Root span: Overall agent invocation
- LLM spans: Foundation model invocations
- Action group spans: Tool/action executions
- Knowledge base spans: Retrievals from knowledge bases
- Guardrail spans: Content safety and validation checks

Each span includes attributes like:
- Input/output prompts and completions
- Token usage metrics
- Duration timing
- Session metadata

#### Customizing the Integration

To modify the integration:

1. **Span Attributes**: Extend `SpanAttributes` in `constants.py` to define new attributes:
   ```python
   class SpanAttributes:
       # Add custom attributes
       CUSTOM_METRIC = "your.custom.metric"
   ```

2. **Custom Spans**: Modify or create handlers in `handlers.py` to track additional operations:
   ```python
   def handle_custom_operation(trace_data, parent_span):
       # Create a span for your custom operation
       span = tracer.start_span(name="custom_operation", context=trace.set_span_in_context(parent_span))
   ```

3. **Span Processing**: Adjust `process_trace_event` in `agent.py` to recognize new trace types:
   ```python
   def process_trace_event(trace_data, parent_span):
       # Add handler for your custom trace type
       if "customTrace" in trace:
           handle_custom_operation(trace_data, parent_span)
   ```

4. **Export Configuration**: Change the OTLP export configuration for different targets:
   ```python
   # To use a different trace backend
   os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://your-custom-endpoint"
   ```
