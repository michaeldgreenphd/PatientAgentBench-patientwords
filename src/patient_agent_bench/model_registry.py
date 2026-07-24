# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Bedrock Model Registry for PatientAgentBench.

Provides a centralized registry of supported Bedrock models with their
default parameters. Users can reference models by simple names in config
files, or provide custom model specifications.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ModelCapability(Enum):
    """Model capabilities for filtering."""
    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"


@dataclass(frozen=True)
class ModelSpec:
    """
    Specification for a model.

    Attributes:
        model_id: Full model ID (Bedrock model ID, OpenAI model name, Mantle model id, etc.)
        display_name: Human-readable name
        provider: Access CHANNEL — how we call the model. One of:
            "bedrock"                → AWS Bedrock Converse (boto3 IAM)
            "openai-protocol-api"    → OpenAI-compatible HTTP endpoint (ChatOpenAI).
                                       Covers OpenAI direct, LiteLLM proxies, AND the
                                       AWS Bedrock Mantle gateway. Distinguished by `auth`.
            "anthropic-protocol-api" → Anthropic-compatible HTTP endpoint
                                       (ChatAnthropic). Covers Anthropic direct AND
                                       the Mantle gateway. Distinguished by `auth`.
        developer: Who BUILT the model — the vendor (informational + used for
            channel-specific feature routing). E.g. "anthropic", "openai",
            "mistral", "amazon", "google", "moonshot", "deepseek", "qwen", "meta".
        auth: For the *-protocol-api channels only — how to authenticate:
            "api_key" → API key header (OpenAI/Anthropic direct, LiteLLM)
            "sigv4"   → SigV4 signing against the bedrock-mantle service (Mantle)
        default_temperature: Recommended temperature for this model
        default_max_tokens: Recommended max tokens
        capabilities: List of model capabilities
        description: Brief description of the model
        notes: Additional notes (e.g., limitations)
        api_prefix: For provider="openai-protocol-api" with auth="sigv4" (Mantle)
            only — the URL path family on the Mantle host. "/v1" for
            chat-completions models (default), "/openai/v1" for the OpenAI
            Responses API (reasoning models).
        use_responses_api: When True, call the OpenAI Responses API instead of
            Chat Completions (ChatOpenAI flag). Used by reasoning models.
        base_url_env: Optional name of an env var holding the OpenAI-compatible
            base URL. Only meaningful for provider="openai-protocol-api" with
            auth="api_key" (e.g. a LiteLLM proxy). When unset, ChatOpenAI uses
            its default (api.openai.com).
        api_key_env: Optional name of an env var holding the API key. Only
            meaningful for auth="api_key". When unset, falls back to OPENAI_API_KEY.
        input_price_per_1m: On-demand list price in USD per 1,000,000 input
            tokens. None → unpriced; cost renders as N/A (never guessed).
        output_price_per_1m: On-demand list price in USD per 1,000,000 output
            tokens. None → unpriced.
    """
    model_id: str
    display_name: str
    provider: str          # access channel: "bedrock" | "openai-protocol-api"
    developer: str = ""    # model vendor: "anthropic" | "openai" | "mistral" | ...
    auth: str = ""         # for openai-protocol-api: "api_key" | "sigv4"
    default_temperature: Optional[float] = 0  # None → omit `temperature` from the
                                              # request entirely (e.g. Claude Sonnet 5
                                              # rejects it with a ValidationException)
    default_max_tokens: int = 4096
    capabilities: tuple = field(default=(ModelCapability.TEXT, ModelCapability.TOOL_USE))
    description: str = ""
    notes: str = ""
    api_prefix: str = "/v1"
    use_responses_api: bool = False
    base_url_env: Optional[str] = None
    api_key_env: Optional[str] = None
    # On-demand list price, USD per 1M tokens (input, output). None → unpriced.
    input_price_per_1m: Optional[float] = None
    output_price_per_1m: Optional[float] = None


# =============================================================================
# Model Registry
# =============================================================================

MODEL_STORE: Dict[str, ModelSpec] = {
    # -------------------------------------------------------------------------
    # Anthropic Claude Models (using global inference profiles for better availability)
    # -------------------------------------------------------------------------
    "claude-opus-4.5-bedrock": ModelSpec(
        model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
        display_name="Claude Opus 4.5",
        provider="bedrock",
        developer="anthropic",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Most capable Claude model, best for complex reasoning and analysis",
        input_price_per_1m=5.00,
        output_price_per_1m=25.00,
    ),
    "claude-opus-4.8-bedrock": ModelSpec(
        model_id="global.anthropic.claude-opus-4-8",
        display_name="Claude Opus 4.8",
        provider="bedrock",
        developer="anthropic",
        default_temperature=None,  # Bedrock rejects `temperature` for this model
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Latest Opus model, most capable for complex reasoning and agentic tasks",
        input_price_per_1m=5.00,
        output_price_per_1m=25.00,
    ),
    "claude-opus-4.6-bedrock": ModelSpec(
        model_id="global.anthropic.claude-opus-4-6-v1",
        display_name="Claude Opus 4.6",
        provider="bedrock",
        developer="anthropic",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Opus model, strong for autonomous coding and complex agents",
        input_price_per_1m=5.00,
        output_price_per_1m=25.00,
    ),
    "claude-sonnet-4.6-bedrock": ModelSpec(
        model_id="global.anthropic.claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        provider="bedrock",
        developer="anthropic",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Latest Sonnet model, balanced performance with improved reasoning",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
    ),
    "claude-sonnet-4.5-bedrock": ModelSpec(
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        display_name="Claude Sonnet 4.5",
        provider="bedrock",
        developer="anthropic",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Balanced performance and cost, good for most tasks",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
    ),
    "claude-sonnet-4.0-bedrock": ModelSpec(
        model_id="global.anthropic.claude-sonnet-4-20250514-v1:0",
        display_name="Claude Sonnet 4",
        provider="bedrock",
        developer="anthropic",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Efficient hybrid reasoning model, good for high-volume production tasks",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
    ),
    "claude-haiku-4.5-bedrock": ModelSpec(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        display_name="Claude Haiku 4.5",
        provider="bedrock",
        developer="anthropic",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Fastest Claude model, optimized for speed and efficiency",
        input_price_per_1m=1.00,
        output_price_per_1m=5.00,
    ),
    "claude-sonnet-5-bedrock": ModelSpec(
        model_id="global.anthropic.claude-sonnet-5",
        display_name="Claude Sonnet 5",
        provider="bedrock",
        developer="anthropic",
        default_temperature=None,  # Bedrock rejects `temperature` for this model
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Latest Sonnet model, strongest balanced reasoning and agentic performance",
        input_price_per_1m=2.00,
        output_price_per_1m=10.00,
    ),

    # Claude via the direct Anthropic API (auth="api_key", ANTHROPIC_API_KEY).
    # model_id is the Anthropic API model name, not a Bedrock inference-profile id.
    "claude-opus-4.8-api": ModelSpec(
        model_id="claude-opus-4-8",
        display_name="Claude Opus 4.8 (Anthropic API)",
        provider="anthropic-protocol-api",
        developer="anthropic",
        auth="api_key",
        default_temperature=None,  # Opus 4.8 rejects the `temperature` param
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Claude Opus 4.8 via the direct Anthropic API",
    ),
    "claude-sonnet-5-api": ModelSpec(
        model_id="claude-sonnet-5",
        display_name="Claude Sonnet 5 (Anthropic API)",
        provider="anthropic-protocol-api",
        developer="anthropic",
        auth="api_key",
        default_temperature=None,  # Sonnet 5 rejects the `temperature` param
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Claude Sonnet 5 via the direct Anthropic API",
    ),
    "claude-sonnet-4.6-api": ModelSpec(
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6 (Anthropic API)",
        provider="anthropic-protocol-api",
        developer="anthropic",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Claude Sonnet 4.6 via the direct Anthropic API",
    ),
    "claude-haiku-4.5-api": ModelSpec(
        model_id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5 (Anthropic API)",
        provider="anthropic-protocol-api",
        developer="anthropic",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="Claude Haiku 4.5 via the direct Anthropic API",
    ),

    # -------------------------------------------------------------------------
    # Amazon Nova Models (using global inference profiles where available)
    # -------------------------------------------------------------------------
    "nova-premier-bedrock": ModelSpec(
        model_id="us.amazon.nova-premier-v1:0",
        display_name="Nova Premier",
        provider="bedrock",
        developer="amazon",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Amazon's most capable Nova model",
        notes="Using US profile (global not available)",
    ),
    "nova-pro-bedrock": ModelSpec(
        model_id="us.amazon.nova-pro-v1:0",
        display_name="Nova Pro",
        provider="bedrock",
        developer="amazon",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Amazon's balanced Nova model",
        notes="Using US profile (global not available)",
    ),
    "nova-lite-bedrock": ModelSpec(
        model_id="global.amazon.nova-2-lite-v1:0",
        display_name="Nova 2 Lite",
        provider="bedrock",
        developer="amazon",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Amazon's lightweight Nova model",
        input_price_per_1m=0.30,
        output_price_per_1m=2.50,
    ),

    # -------------------------------------------------------------------------
    # DeepSeek Models (using inference profiles where available)
    # -------------------------------------------------------------------------
    "deepseek-v3-bedrock": ModelSpec(
        model_id="deepseek.v3-v1:0",
        display_name="DeepSeek V3",
        provider="bedrock",
        developer="deepseek",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="DeepSeek's latest general-purpose model",
        notes="No image support, no inference profile available",
        input_price_per_1m=0.58,
        output_price_per_1m=1.68,
    ),
    "deepseek-r1-bedrock": ModelSpec(
        model_id="us.deepseek.r1-v1:0",
        display_name="DeepSeek R1",
        provider="bedrock",
        developer="deepseek",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="DeepSeek reasoning model optimized for complex tasks",
        notes="No image support, using US profile (global not available)",
        input_price_per_1m=1.35,
        output_price_per_1m=5.40,
    ),

    # -------------------------------------------------------------------------
    # Qwen Models (direct model IDs - no inference profiles available)
    # -------------------------------------------------------------------------
    "qwen3-235b-bedrock": ModelSpec(
        model_id="qwen.qwen3-235b-a22b-2507-v1:0",
        display_name="Qwen3 235B A22B",
        provider="bedrock",
        developer="qwen",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Qwen's largest model with 235B parameters",
        input_price_per_1m=0.22,
        output_price_per_1m=0.88,
    ),
    "qwen3-next-80b-bedrock": ModelSpec(
        model_id="qwen.qwen3-next-80b-a3b",
        display_name="Qwen3 Next 80B A3B",
        provider="bedrock",
        developer="qwen",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="Qwen's efficient 80B parameter model",
        notes="No image support",
        input_price_per_1m=0.15,
        output_price_per_1m=1.20,
    ),
    "qwen3-32b-bedrock": ModelSpec(
        model_id="qwen.qwen3-32b-v1:0",
        display_name="Qwen3 32B",
        provider="bedrock",
        developer="qwen",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="Qwen's dense 32B parameter model",
        # TEXT-only, ON_DEMAND (direct model id, no inference profile).
        notes="No image support",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),

    # -------------------------------------------------------------------------
    # Meta Llama Models (using US inference profiles - required for on-demand)
    # -------------------------------------------------------------------------
    "llama3-2-90b-bedrock": ModelSpec(
        model_id="us.meta.llama3-2-90b-instruct-v1:0",
        display_name="Llama 3.2 90B Instruct",
        provider="bedrock",
        developer="meta",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Llama 3.2 largest model with vision support",
        notes="Using US inference profile",
        input_price_per_1m=0.72,
        output_price_per_1m=0.72,
    ),
    "llama3-3-70b-bedrock": ModelSpec(
        model_id="us.meta.llama3-3-70b-instruct-v1:0",
        display_name="Llama 3.3 70B Instruct",
        provider="bedrock",
        developer="meta",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="Llama 3.3 latest text model, strong reasoning",
        notes="Using US inference profile",
        input_price_per_1m=0.72,
        output_price_per_1m=0.72,
    ),
    "llama4-scout-bedrock": ModelSpec(
        model_id="us.meta.llama4-scout-17b-instruct-v1:0",
        display_name="Llama 4 Scout 17B Instruct",
        provider="bedrock",
        developer="meta",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="Llama 4 Scout - efficient next-gen model",
        notes="Using US inference profile",
        input_price_per_1m=0.17,
        output_price_per_1m=0.66,
    ),
    "llama4-maverick-bedrock": ModelSpec(
        model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
        display_name="Llama 4 Maverick 17B Instruct",
        provider="bedrock",
        developer="meta",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="Llama 4 Maverick - creative next-gen model",
        notes="Using US inference profile",
        input_price_per_1m=0.24,
        output_price_per_1m=0.97,
    ),

    # -------------------------------------------------------------------------
    # GPT-OSS Models (Bedrock-served, direct model IDs)
    # -------------------------------------------------------------------------
    "gpt-oss-120b-bedrock": ModelSpec(
        model_id="openai.gpt-oss-120b-1:0",
        display_name="GPT-OSS 120B",
        provider="bedrock",
        developer="openai",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="OpenAI's GPT-OSS 120B model",
        notes="No image support",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    "gpt-oss-safeguard-120b-bedrock": ModelSpec(
        model_id="openai.gpt-oss-safeguard-120b",
        display_name="GPT-OSS Safeguard 120B",
        provider="bedrock",
        developer="openai",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="OpenAI's GPT-OSS 120B with enhanced safety guardrails",
        notes="No image support",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),

    # -------------------------------------------------------------------------
    # OpenAI API Models (direct OpenAI API, requires OPENAI_API_KEY)
    # GPT-5 family, GPT-5.2 family, and reasoning models
    # -------------------------------------------------------------------------
    "gpt-5-api": ModelSpec(
        model_id="gpt-5",
        display_name="GPT-5",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(
            ModelCapability.TEXT,
            ModelCapability.IMAGE,
            ModelCapability.TOOL_USE,
        ),
        description="OpenAI's GPT-5 flagship model via API",
    ),
    "gpt-5-mini-api": ModelSpec(
        model_id="gpt-5-mini",
        display_name="GPT-5 Mini",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(
            ModelCapability.TEXT,
            ModelCapability.IMAGE,
            ModelCapability.TOOL_USE,
        ),
        description="OpenAI's GPT-5 Mini via API, lower cost",
    ),
    "gpt-5-nano-api": ModelSpec(
        model_id="gpt-5-nano",
        display_name="GPT-5 Nano",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(
            ModelCapability.TEXT,
            ModelCapability.TOOL_USE,
        ),
        description="OpenAI's GPT-5 Nano via API, high throughput",
    ),
    "gpt-5.2-api": ModelSpec(
        model_id="gpt-5.2",
        display_name="GPT-5.2",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(
            ModelCapability.TEXT,
            ModelCapability.IMAGE,
            ModelCapability.TOOL_USE,
        ),
        description="OpenAI's GPT-5.2 via API, complex enterprise tasks",
    ),
    "gpt-5.2-pro-api": ModelSpec(
        model_id="gpt-5.2-pro",
        display_name="GPT-5.2 Pro",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(
            ModelCapability.TEXT,
            ModelCapability.IMAGE,
            ModelCapability.TOOL_USE,
        ),
        description="OpenAI's GPT-5.2 Pro via API, more compute",
    ),
    "gpt-5.2-chat-latest-api": ModelSpec(
        model_id="gpt-5.2-chat-latest",
        display_name="GPT-5.2 Chat Latest",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=1,
        default_max_tokens=4096,
        capabilities=(
            ModelCapability.TEXT,
            ModelCapability.IMAGE,
            ModelCapability.TOOL_USE,
        ),
        description="OpenAI's GPT-5.2 ChatGPT model via API",
        input_price_per_1m=1.75,
        output_price_per_1m=14.00,
    ),
    "gpt-5.4-api": ModelSpec(
        model_id="gpt-5.4",
        display_name="GPT-5.4 (OpenAI API)",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        use_responses_api=True,
        default_temperature=1,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="OpenAI GPT-5.4 directly via the OpenAI API (Responses API)",
    ),
    "gpt-5.5-api": ModelSpec(
        model_id="gpt-5.5",
        display_name="GPT-5.5 (OpenAI API)",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        use_responses_api=True,
        default_temperature=1,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="OpenAI GPT-5.5 directly via the OpenAI API (Responses API)",
    ),
    "o3-api": ModelSpec(
        model_id="o3",
        display_name="o3",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="OpenAI's o3 reasoning model via OpenAI API",
    ),
    "o3-mini-api": ModelSpec(
        model_id="o3-mini",
        display_name="o3 Mini",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="OpenAI's o3-mini reasoning model via OpenAI API",
    ),
    "o4-mini-api": ModelSpec(
        model_id="o4-mini",
        display_name="o4 Mini",
        provider="openai-protocol-api",
        developer="openai",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="OpenAI's o4-mini reasoning model via OpenAI API",
    ),

    # -------------------------------------------------------------------------
    # LiteLLM Proxy Models (OpenAI-compatible endpoint)
    # Set LITELLM_BASE_URL and LITELLM_API_KEY in .env. Works with any
    # LiteLLM-based proxy that exposes /v1/chat/completions. The model_id
    # below must match a model registered in the proxy's config.
    # -------------------------------------------------------------------------
    "gemini-3.1-pro-api": ModelSpec(
        model_id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro",
        provider="openai-protocol-api",
        developer="google",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Google Gemini 3.1 Pro routed via a LiteLLM proxy",
        base_url_env="LITELLM_BASE_URL",
        api_key_env="LITELLM_API_KEY",
        # Standard-context pricing (prompts <= 200K); long-context surcharge
        # (>200K) not applied — our peak inputs are far below.
        input_price_per_1m=2.00,
        output_price_per_1m=12.00,
    ),
    "gemini-3-flash-api": ModelSpec(
        model_id="gemini-3-flash-preview",
        display_name="Gemini 3 Flash",
        provider="openai-protocol-api",
        developer="google",
        auth="api_key",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="Google Gemini 3 Flash routed via a LiteLLM proxy",
        base_url_env="LITELLM_BASE_URL",
        api_key_env="LITELLM_API_KEY",
        input_price_per_1m=0.50,
        output_price_per_1m=3.00,
    ),

    # -------------------------------------------------------------------------
    # AWS Bedrock Mantle Gateway (OpenAI protocol, SigV4 auth)
    # Endpoint: https://bedrock-mantle.{AWS_REGION}.api.aws{api_prefix}
    # Reasoning models (GPT-5.x) use the Responses API at /openai/v1.
    # Chat models use /v1/chat/completions.
    # -------------------------------------------------------------------------
    "gpt-5.4-mantle": ModelSpec(
        model_id="openai.gpt-5.4",
        display_name="GPT-5.4",
        provider="openai-protocol-api",
        developer="openai",
        auth="sigv4",
        api_prefix="/openai/v1",
        use_responses_api=True,
        default_temperature=1,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="OpenAI GPT-5.4 via Bedrock Mantle (Responses API)",
        input_price_per_1m=2.75,
        output_price_per_1m=16.50,
    ),
    "gpt-5.4-mantle-high": ModelSpec(
        model_id="openai.gpt-5.4",
        display_name="GPT-5.4 (high reasoning)",
        provider="openai-protocol-api",
        developer="openai",
        auth="sigv4",
        api_prefix="/openai/v1",
        use_responses_api=True,
        default_temperature=1,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="OpenAI GPT-5.4 via Bedrock Mantle with reasoning_effort=high",
        notes="Set reasoning_effort='high' on the ModelConfig to pin reasoning_effort.",
        input_price_per_1m=2.75,
        output_price_per_1m=16.50,
    ),
    "gpt-5.5-mantle": ModelSpec(
        model_id="openai.gpt-5.5",
        display_name="GPT-5.5",
        provider="openai-protocol-api",
        developer="openai",
        auth="sigv4",
        api_prefix="/openai/v1",
        use_responses_api=True,
        default_temperature=1,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="OpenAI GPT-5.5 via Bedrock Mantle (Responses API)",
        input_price_per_1m=5.50,
        output_price_per_1m=33.00,
    ),
    "gpt-5.5-mantle-high": ModelSpec(
        model_id="openai.gpt-5.5",
        display_name="GPT-5.5 (high reasoning)",
        provider="openai-protocol-api",
        developer="openai",
        auth="sigv4",
        api_prefix="/openai/v1",
        use_responses_api=True,
        default_temperature=1,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.TOOL_USE),
        description="OpenAI GPT-5.5 via Bedrock Mantle with reasoning_effort=high",
        notes="Set reasoning_effort='high' on the ModelConfig to pin reasoning_effort.",
        input_price_per_1m=5.50,
        output_price_per_1m=33.00,
    ),
    "mistral-large-3-mantle": ModelSpec(
        model_id="mistral.mistral-large-3-675b-instruct",
        display_name="Mistral Large 3",
        provider="openai-protocol-api",
        developer="mistral",
        auth="sigv4",
        api_prefix="/v1",
        default_temperature=0,
        default_max_tokens=4096,
        capabilities=(ModelCapability.TEXT, ModelCapability.TOOL_USE),
        description="Mistral Large 3 (675B) via Bedrock Mantle (Chat Completions)",
    ),
}


def get_model_spec(model_name: str) -> Optional[ModelSpec]:
    """
    Get model specification by registry key.

    Registry keys follow the ``family-version-provider`` convention, where the
    provider suffix is one of ``-bedrock`` (Bedrock Converse), ``-api`` (a direct
    vendor API or an OpenAI-compatible endpoint — OpenAI/Anthropic/LiteLLM), or
    ``-mantle`` (the Bedrock Mantle SigV4 gateway). Examples:
    ``claude-sonnet-5-bedrock``, ``claude-sonnet-5-api``, ``gpt-5.5-api``,
    ``gpt-5.5-mantle``, ``nova-premier-bedrock``.

    Args:
        model_name: Registry key.

    Returns:
        ModelSpec if found, None otherwise.
    """
    return MODEL_STORE.get(model_name)


def get_model_pricing(model_name: str) -> Optional[dict]:
    """Return {input_price_per_1m, output_price_per_1m} for a model, or None.

    Resolves ``model_name`` (accepts a registry key or a full model_id — run
    configs may carry only the model_id) to its ModelSpec and reads the price
    fields off the spec. Returns None when the model is unknown or unpriced, so
    callers render cost as N/A.
    """
    spec = get_model_spec(model_name)
    if spec is None:
        # Fall back to matching by full model_id.
        for s in MODEL_STORE.values():
            if s.model_id == model_name:
                spec = s
                break
    if spec is None:
        return None
    if spec.input_price_per_1m is None or spec.output_price_per_1m is None:
        return None
    return {
        "input_price_per_1m": spec.input_price_per_1m,
        "output_price_per_1m": spec.output_price_per_1m,
    }



def list_models(
    provider: Optional[str] = None,
    capability: Optional[ModelCapability] = None,
) -> List[str]:
    """
    List available models, optionally filtered.

    Args:
        provider: Filter by provider (anthropic, amazon, deepseek, qwen)
        capability: Filter by capability (TEXT, IMAGE, TOOL_USE)

    Returns:
        List of model names
    """
    models = []
    for name, spec in MODEL_STORE.items():
        if provider and spec.provider != provider:
            continue
        if capability and capability not in spec.capabilities:
            continue
        models.append(name)
    return models


def list_models_with_image_support() -> List[str]:
    """List models that support image input."""
    return list_models(capability=ModelCapability.IMAGE)


def list_models_text_only() -> List[str]:
    """List models that only support text (no image)."""
    return [
        name for name, spec in MODEL_STORE.items()
        if ModelCapability.IMAGE not in spec.capabilities
    ]


def print_model_catalog() -> str:
    """
    Generate a formatted catalog of all available models.

    Returns:
        Formatted string with model information
    """
    lines = ["Available Bedrock Models:", "=" * 60]

    # Group by provider
    providers = {}
    for name, spec in MODEL_STORE.items():
        if spec.provider not in providers:
            providers[spec.provider] = []
        providers[spec.provider].append((name, spec))

    for provider, models in sorted(providers.items()):
        lines.append(f"\n{provider.upper()}")
        lines.append("-" * 40)
        for name, spec in models:
            image_support = "✓" if ModelCapability.IMAGE in spec.capabilities else "✗"
            lines.append(f"  {name}")
            lines.append(f"    Display: {spec.display_name}")
            lines.append(f"    Model ID: {spec.model_id}")
            lines.append(f"    Image Support: {image_support}")
            if spec.notes:
                lines.append(f"    Notes: {spec.notes}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
