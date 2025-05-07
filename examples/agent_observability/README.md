# Amazon Bedrock Agent Observability Tools
This repository contains two implementations for instrumenting AWS Bedrock Agents to provide comprehensive observability:

> OpenInference Instrumentation: A framework that leverages OpenInference semantic conventions for tracing Bedrock Agent invocations.

> OpenTelemetry Instrumentation: A complete observability solution that follows OpenTelemetry standards to send trace data to any compatible platform.

## Purpose
These tools enable developers to gain visibility into their Bedrock Agent interactions, helping with:

- Debugging complex agent behaviors
- Understanding performance patterns and bottlenecks
- Monitoring token usage and costs
- Analyzing the flow of information through agents

### Important Note
The field of GenAI observability is rapidly evolving. These implementations combine established GenAI observability semantics with Amazon Bedrock Agent-specific semantics. They will be updated as more standardized semantics become available in the industry.

## Repository Structure

- /open-inference-instrumentation: Implementation using OpenInference semantic conventions

- /opentelemetry-instrumentation: Implementation using OpenTelemetry standards

Each subdirectory contains its own detailed README with specific setup instructions and usage examples.

### Key Features of Both Implementations
Complete span hierarchy with proper parent-child relationships
Token usage tracking for LLM operations
Support for both streaming and non-streaming responses
Compatible with multiple observability platforms (Arize, Langfuse, Weave etc.)
Detailed trace and performance metrics

## Contributing
Contributions are welcome as we continue to evolve these tools alongside emerging industry standards for GenAI observability.