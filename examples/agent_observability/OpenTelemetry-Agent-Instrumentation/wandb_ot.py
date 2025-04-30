import json, os
import base64
import openai
from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WANDB_BASE_URL = "https://trace.wandb.ai"
PROJECT_ID = "capecape/otel_bedrock"

OTEL_EXPORTER_OTLP_ENDPOINT = f"{WANDB_BASE_URL}/otel/v1/traces"

# Can be found at https://wandb.ai/authorize
WANDB_API_KEY = "0d34f7c7d794b81ffaca89c94a328a121d749158"
AUTH = base64.b64encode(f"api:{WANDB_API_KEY}".encode()).decode()

OTEL_EXPORTER_OTLP_HEADERS = {
    "Authorization": f"Basic {AUTH}",
    "project_id": PROJECT_ID,
}

tracer_provider = trace_sdk.TracerProvider()

# Configure the OTLP exporter
exporter = OTLPSpanExporter(
    endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
    headers=OTEL_EXPORTER_OTLP_HEADERS,
)

# Add the exporter to the tracer provider
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

# Optionally, print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(tracer_provider)
# Creates a tracer from the global tracer provider
tracer = trace.get_tracer(__name__)
tracer.start_span('name=standard-span')

def my_function():
    with tracer.start_as_current_span("outer_span") as outer_span:
        client = openai.OpenAI()
        input_messages=[{"role": "user", "content": "Describe OTEL in a single sentence."}]
        # This will only appear in the side panel
        outer_span.set_attribute("input.value", json.dumps(input_messages))
        # This follows conventions and will appear in the dashboard
        outer_span.set_attribute("gen_ai.system", 'openai')
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=input_messages,
            max_tokens=20,
            stream=True,
            stream_options={"include_usage": True},
        )
        out = ""
        for chunk in response:
            if chunk.choices and (content := chunk.choices[0].delta.content):
                print(content)
                out += content
        # This will only appear in the side panel
        outer_span.set_attribute("output.value", json.dumps({"content": out}))

if __name__ == "__main__":
    my_function()