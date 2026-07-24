# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Configuration management for PatientAgentBench.

Provides centralized configuration for AWS Bedrock, model settings, and other parameters.

AWS Credentials Resolution (standard boto3 credential chain):
1. .env file / environment variables (explicit configuration)
2. Shared config/credentials (~/.aws/credentials, e.g. 'aws configure')
3. IAM instance role (EC2/Lambda/ECS - lowest priority)

Auto-authentication:
- If credentials are expired, refresh is delegated to refresh_credentials_hook().
  In this release that hook is a no-op and the ambient AWS credentials are used.
"""

import importlib
import os
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import boto3
import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
from dotenv import load_dotenv

from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from patient_agent_bench.logging_config import get_logger

# Bedrock Mantle: SigV4 is signed against this service name (not "bedrock").
MANTLE_SIGV4_SERVICE = "bedrock-mantle"


class _MantleSigV4Auth(httpx.Auth):
    """httpx auth hook that SigV4-signs each request for Bedrock Mantle.

    Mirrors create_bedrock_client_with_role(): if a role_arn is provided it
    assumes that role and signs with the temporary credentials; otherwise it
    signs with the ambient/base credentials. Credentials are resolved on every
    request so refresh and assumed-role rotation are picked up automatically
    for long-running benchmarks (falls back to base creds if assume fails).
    """

    def __init__(self, region: str, role_arn: Optional[str] = None,
                 session_name: str = "patient-agent-bench-mantle"):
        self._region = region
        self._role_arn = role_arn
        self._session_name = session_name

    def _resolve_credentials(self):
        if not self._role_arn:
            return boto3.Session().get_credentials()
        try:
            resp = boto3.client("sts").assume_role(
                RoleArn=self._role_arn,
                RoleSessionName=self._session_name,
                DurationSeconds=3600,
            )
            c = resp["Credentials"]
            return boto3.Session(
                aws_access_key_id=c["AccessKeyId"],
                aws_secret_access_key=c["SecretAccessKey"],
                aws_session_token=c["SessionToken"],
            ).get_credentials()
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Mantle: failed to assume role %s: %s. Using base credentials.",
                self._role_arn, e,
            )
            return boto3.Session().get_credentials()

    def auth_flow(self, request):
        creds = self._resolve_credentials()
        aws_req = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers={"Content-Type": "application/json"},
        )
        SigV4Auth(creds.get_frozen_credentials(), MANTLE_SIGV4_SERVICE, self._region).add_auth(aws_req)
        for k, v in aws_req.headers.items():
            request.headers[k] = v
        yield request
from patient_agent_bench.model_registry import get_model_spec

logger = get_logger(__name__)


def load_prompt(agent_type: str, prompt_name: str) -> str:
    """
    Load a prompt template from the agent's directory or an absolute file path.

    Args:
        agent_type: Either "user_agent" or "assistant_agent"
        prompt_name: Name of the prompt file (without .py extension) OR an absolute file path

    Returns:
        The SYSTEM_PROMPT string from the module, or file contents if absolute path

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        AttributeError: If the module doesn't export SYSTEM_PROMPT (for module loading)
    """
    # Check if prompt_name is an absolute path - load directly from file
    if os.path.isabs(prompt_name):
        if not os.path.exists(prompt_name):
            raise FileNotFoundError(f"Prompt file not found: {prompt_name}")
        with open(prompt_name, "r", encoding="utf-8") as f:
            return f.read()

    # Module-based loading for relative prompt names
    if agent_type not in ("user_agent", "assistant_agent"):
        raise ValueError(
            f"Invalid agent_type '{agent_type}'. Must be 'user_agent' or 'assistant_agent'."
        )

    module_path = f"patient_agent_bench.{agent_type}.{prompt_name}"

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # Construct the expected file path for a clear error message
        expected_path = f"src/patient_agent_bench/{agent_type}/{prompt_name}.py"
        raise FileNotFoundError(
            f"Prompt file not found: {expected_path}. "
            f"Create a file with a SYSTEM_PROMPT constant."
        ) from exc

    if not hasattr(module, "SYSTEM_PROMPT"):
        raise AttributeError(
            f"Module '{module_path}' does not export SYSTEM_PROMPT constant."
        )

    prompt: str = module.SYSTEM_PROMPT
    return prompt


def format_prompt_safe(template: str, **kwargs) -> str:
    """
    Format a prompt template, ignoring missing placeholders.

    Logs a warning for any expected placeholders not found in the template.
    Only formats placeholders that exist in the template.

    Args:
        template: The prompt template string with {placeholder} syntax
        **kwargs: Key-value pairs for placeholder substitution

    Returns:
        The formatted prompt string
    """
    # Find all placeholders in the template
    template_placeholders = set(re.findall(r"\{(\w+)\}", template))

    # Warn about expected placeholders not found in template
    for key in kwargs:
        if key not in template_placeholders:
            logger.warning("Placeholder '{%s}' not found in prompt template", key)

    # Only substitute placeholders that exist in the template
    result = template
    for key, value in kwargs.items():
        if key in template_placeholders:
            result = result.replace(f"{{{key}}}", str(value))

    return result

# Load .env file if it exists (override=True means .env takes highest priority)
_env_file = Path(__file__).parent.parent.parent.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=True)
else:
    # Also check current working directory
    load_dotenv(override=True)


@dataclass
class RolePoolManager:
    """
    Manages a pool of AWS ARN roles for load distribution.

    Supports round-robin distribution of roles to parallel workers.
    Thread-safe for concurrent access.
    """

    roles: List[str] = field(default_factory=list)
    _index: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls) -> "RolePoolManager":
        """
        Load roles from AWS_ARN_ROLE environment variable.

        Supports:
        - Single role: "arn:aws:iam::123:role/my-role"
        - Multiple roles: "arn:aws:iam::123:role/role1,arn:aws:iam::456:role/role2"

        Returns:
            RolePoolManager with parsed roles, or empty pool if env var not set.
        """
        env_value = os.environ.get("AWS_ARN_ROLE", "")
        if not env_value:
            return cls(roles=[])

        # Split by comma and strip whitespace, filter empty strings
        roles = [r.strip() for r in env_value.split(",") if r.strip()]
        return cls(roles=roles)

    def get_role(self) -> Optional[str]:
        """
        Get next role using round-robin distribution.

        Thread-safe index increment ensures even distribution across workers.

        Returns:
            Next role ARN, or None if pool is empty.
        """
        if not self.roles:
            return None

        with self._lock:
            role = self.roles[self._index % len(self.roles)]
            self._index += 1
            return role

    def get_random_role(self) -> Optional[str]:
        """
        Get a random role from the pool.

        Returns:
            Random role ARN, or None if pool is empty.
        """
        if not self.roles:
            return None
        return random.choice(self.roles)

    def __len__(self) -> int:
        """Return the number of roles in the pool."""
        return len(self.roles)


def check_credentials_valid() -> bool:
    """
    Check whether usable AWS credentials are available.

    Resolves credentials through the standard boto3 chain and confirms they are
    not expired by making a single STS GetCallerIdentity call.

    Returns True if the call succeeds, False otherwise.
    """
    try:
        boto3.client('sts').get_caller_identity()
        return True
    except Exception as e:
        logger.debug("Credentials check failed: %s", e)
        return False


def refresh_credentials_hook() -> bool:
    """
    Refresh AWS credentials via an external credential helper.

    This is a no-op hook in the public release: it performs no credential
    refresh and lets the application proceed with the ambient AWS credentials
    already available to boto3 (environment variables, shared config/profile,
    instance/role metadata, etc.). Deployments that need automated credential
    refresh can override this hook with their own implementation.
    """
    logger.debug(
        "no credential refresh hook configured; using ambient AWS credentials"
    )
    return False


def create_bedrock_client_with_role(
    role_arn: Optional[str] = None,
    region: Optional[str] = None,
    session_name: str = "patient-agent-bench",
    read_timeout: int = 900,
):
    """
    Create a boto3 bedrock-runtime client, optionally with assumed role credentials.

    This function creates an isolated client with its own credentials, avoiding
    race conditions when used in parallel tasks. Each task gets its own client
    with the appropriate role credentials.

    Ensures base credentials are valid before attempting role assumption.

    Args:
        role_arn: Optional AWS IAM role ARN to assume. If None, uses default credentials.
        region: AWS region. If None, uses AWS_REGION env var.
        session_name: Session name for assumed role (default: "patient-agent-bench")
        read_timeout: Read timeout in seconds (default: 900 for long LLM responses)

    Returns:
        boto3 bedrock-runtime client with appropriate credentials

    Raises:
        RuntimeError: If credentials cannot be obtained after refresh attempt.
    """
    region = region or os.environ.get("AWS_REGION")

    # Configure longer timeout for LLM responses (especially with high max_tokens)
    client_config = Config(
        read_timeout=read_timeout,
        retries={"max_attempts": 0},  # Let our retry logic handle retries
    )

    # Ensure base credentials are valid before creating any client
    if not check_credentials_valid():
        logger.info("Base credentials invalid, refreshing before creating bedrock client...")
        if not ensure_credentials():
            raise RuntimeError(
                "Cannot create bedrock client: failed to obtain valid AWS credentials. "
                "Configure AWS credentials (e.g. via environment variables, "
                "'aws configure', or an instance role)."
            )

    if not role_arn:
        # No role to assume, use default credentials
        return boto3.client("bedrock-runtime", region_name=region, config=client_config)

    logger.debug("Creating bedrock client with assumed role: %s", role_arn)

    try:
        # Create STS client with current credentials
        sts = boto3.client("sts")

        # Assume the specified role
        response = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name,
            DurationSeconds=3600,  # 1 hour
        )

        # Extract credentials from response
        credentials = response["Credentials"]

        # Create bedrock client with assumed role credentials
        return boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=client_config,
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )

    except Exception as e:
        logger.warning(
            "Failed to assume role %s: %s. Using base credentials.",
            role_arn,
            e,
        )
        return boto3.client("bedrock-runtime", region_name=region, config=client_config)


def create_chat_model(
    model_config: "ModelConfig",
    bedrock_client=None,
    role_arn: Optional[str] = None,
) -> BaseChatModel:
    """
    Create the appropriate LangChain chat model based on the access channel.

    This is the single factory for all LLM creation in the project. Channels:
      - "openai-protocol-api": ChatOpenAI over an OpenAI-compatible HTTP endpoint.
          auth="api_key" → OpenAI direct or a LiteLLM proxy (API key header).
          auth="sigv4"   → AWS Bedrock Mantle gateway (SigV4 signing).
      - "anthropic-protocol-api": ChatAnthropic over an Anthropic-compatible HTTP
          endpoint.
          auth="api_key" → Anthropic direct (API key header).
          auth="sigv4"   → AWS Bedrock Mantle gateway (SigV4 signing).
      - "bedrock" (default): ChatBedrockConverse (boto3 IAM).

    Args:
        model_config: Model configuration with channel/auth info.
        bedrock_client: boto3 bedrock-runtime client (required for the bedrock
                        channel; ignored otherwise).
        role_arn: Optional IAM role ARN to assume for Mantle SigV4 auth. When
                  None, Mantle signs with base/ambient credentials (mirrors
                  create_bedrock_client_with_role()).

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If required credentials/env vars or the bedrock client are missing.
    """
    if model_config.provider == "openai-protocol-api":
        return _create_openai_protocol_model(model_config, role_arn=role_arn)

    if model_config.provider == "anthropic-protocol-api":
        return _create_anthropic_protocol_model(model_config, role_arn=role_arn)

    # Escape hatch for Bedrock models that don't support the Converse API (e.g.
    # Custom Model Import / imported-model ARNs, which 400 on Converse). Uses
    # LangChain ChatBedrock (InvokeModel under the hood) with the given provider
    # adapter for request/response serialization ("qwen"/"openai" speak the
    # OpenAI chat-completions schema imported vLLM models return). Works for any
    # role; opt-in, so the default path below is unchanged.
    if model_config.bedrock_invoke_provider:
        # Imported models may live in a different region than the run default.
        client = bedrock_client
        if model_config.region:
            client = create_bedrock_client_with_role(
                role_arn, region=model_config.region
            )
        if client is None:
            raise ValueError(
                "bedrock_client (or model_config.region) is required for "
                "bedrock_invoke_provider models."
            )
        model_kwargs = {}
        if model_config.max_tokens is not None:
            model_kwargs["max_tokens"] = model_config.max_tokens
        if model_config.temperature is not None:
            model_kwargs["temperature"] = model_config.temperature
        if model_config.additional_fields:
            model_kwargs.update(model_config.additional_fields)
        return ChatBedrock(
            client=client,
            model_id=model_config.model_id,
            provider=model_config.bedrock_invoke_provider,
            model_kwargs=model_kwargs,
        )

    # Default channel: Bedrock Converse
    # A model-level region override needs its own client, since the caller's
    # client is bound to the run-level region.
    if model_config.region:
        bedrock_client = create_bedrock_client_with_role(
            role_arn, region=model_config.region
        )

    if bedrock_client is None:
        raise ValueError(
            "bedrock_client is required for Bedrock models. "
            "Use create_bedrock_client_with_role() to create one."
        )

    # Extended-thinking is Bedrock-Converse-shaped and only for Anthropic on Bedrock.
    additional_fields = {}
    if (
        model_config.thinking_budget is not None
        and model_config.developer == "anthropic"
        and model_config.provider == "bedrock"
    ):
        additional_fields["thinking"] = {
            "type": "enabled",
            "budget_tokens": int(model_config.thinking_budget),
        }
    if model_config.additional_fields:
        additional_fields.update(model_config.additional_fields)

    return ChatBedrockConverse(
        model=model_config.model_id,
        client=bedrock_client,
        max_tokens=model_config.max_tokens,
        # Omit temperature entirely when None (some models reject the param).
        **({"temperature": model_config.temperature}
           if model_config.temperature is not None else {}),
        **({"additional_model_request_fields": additional_fields}
           if additional_fields else {}),
    )


def _create_openai_protocol_model(
    model_config: "ModelConfig", role_arn: Optional[str] = None
) -> BaseChatModel:
    """Build a ChatOpenAI client for the openai-protocol-api channel.

    Two auth modes:
    - auth="api_key": OpenAI directly (OPENAI_API_KEY, or api_key_env), or any
      OpenAI-compatible endpoint (LiteLLM, vLLM) when base_url_env is set.
    - auth="sigv4": the AWS Bedrock Mantle gateway, which speaks the OpenAI
      protocol; the endpoint is derived from the region and requests are
      SigV4-signed instead of using an API key.
    """
    kwargs = {
        "model": model_config.model_id,
        "max_tokens": model_config.max_tokens,
    }
    # Omit temperature entirely when None (some models reject the param).
    if model_config.temperature is not None:
        kwargs["temperature"] = model_config.temperature
    # reasoning_effort: only set when explicitly pinned (else use model default).
    if model_config.reasoning_effort is not None:
        kwargs["reasoning_effort"] = model_config.reasoning_effort
    if model_config.use_responses_api:
        kwargs["use_responses_api"] = True

    # Pass-through additional_fields as ChatOpenAI kwargs. Applied last so a
    # user-supplied value overrides a named param above. Unknown keys are routed
    # into model_kwargs by langchain-openai, which also works for
    # OpenAI-compatible APIs (LiteLLM, vLLM).
    if model_config.additional_fields:
        kwargs.update(model_config.additional_fields)

    if model_config.auth == "sigv4":
        # Bedrock Mantle gateway: region-derived URL + SigV4 httpx auth.
        # MANTLE_REGION lets Mantle use a different region than AWS_REGION
        # (e.g. Bedrock in us-west-2 but Mantle/GPT-5.x in us-east-1).
        region = os.environ.get("MANTLE_REGION") or os.environ.get("AWS_REGION", "us-east-1")
        host = f"bedrock-mantle.{region}.api.aws"
        prefix = model_config.api_prefix or "/v1"
        http_client = httpx.Client(
            auth=_MantleSigV4Auth(region, role_arn=role_arn),
            timeout=120,  # Mantle has intermittent slow starts
        )
        kwargs["base_url"] = f"https://{host}{prefix}"
        kwargs["api_key"] = "sigv4"  # unused placeholder; real auth is the SigV4 header
        kwargs["http_client"] = http_client
        return ChatOpenAI(**kwargs)

    # auth == "api_key" (OpenAI direct or LiteLLM proxy)
    api_key_env = model_config.api_key_env or "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"{api_key_env} environment variable is required for "
            f"model {model_config.model_id!r}. Set it in your .env file."
        )
    kwargs["api_key"] = api_key
    if model_config.base_url_env:
        base_url = os.environ.get(model_config.base_url_env)
        if not base_url:
            raise ValueError(
                f"{model_config.base_url_env} environment variable is required "
                f"for model {model_config.model_id!r}. Set it in your .env file."
            )
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _create_anthropic_protocol_model(
    model_config: "ModelConfig", role_arn: Optional[str] = None
) -> BaseChatModel:
    """Build a ChatAnthropic client for the anthropic-protocol-api channel.

    Two auth modes:
    - auth="api_key": Anthropic directly (ANTHROPIC_API_KEY, or api_key_env),
      or a custom endpoint when base_url_env is set.
    - auth="sigv4": the AWS Bedrock Mantle gateway, which also speaks the
      Anthropic protocol; the endpoint is derived from the region and requests
      are SigV4-signed instead of using an API key.
    """
    kwargs = {
        "model": model_config.model_id,
        "max_tokens": model_config.max_tokens,
    }
    # Omit temperature entirely when None (some models reject the param).
    if model_config.temperature is not None:
        kwargs["temperature"] = model_config.temperature
    # Reasoning effort (string): Sonnet 5 / Opus 4.7+ adaptive thinking.
    if model_config.reasoning_effort is not None:
        kwargs["reasoning_effort"] = model_config.reasoning_effort
    # Thinking budget (integer): older enabled-thinking with a fixed token budget.
    if model_config.thinking_budget is not None:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": int(model_config.thinking_budget),
        }

    # Pass-through additional_fields as ChatAnthropic kwargs. Applied last so a
    # user-supplied value overrides a named param above. Supports any ChatAnthropic
    # constructor param (e.g. thinking, output_config, anthropic_beta).
    if model_config.additional_fields:
        kwargs.update(model_config.additional_fields)

    if model_config.auth == "sigv4":
        # Bedrock Mantle gateway: region-derived URL + SigV4 httpx auth.
        region = os.environ.get("MANTLE_REGION") or os.environ.get("AWS_REGION", "us-east-1")
        host = f"bedrock-mantle.{region}.api.aws"
        prefix = model_config.api_prefix or "/v1"
        http_client = httpx.Client(
            auth=_MantleSigV4Auth(region, role_arn=role_arn),
            timeout=120,  # Mantle has intermittent slow starts
        )
        kwargs["base_url"] = f"https://{host}{prefix}"
        kwargs["api_key"] = "sigv4"  # unused placeholder; real auth is the SigV4 header
        kwargs["http_client"] = http_client
        return ChatAnthropic(**kwargs)

    # auth == "api_key" (Anthropic direct)
    api_key_env = model_config.api_key_env or "ANTHROPIC_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(
            f"{api_key_env} environment variable is required for "
            f"model {model_config.model_id!r}. Set it in your .env file."
        )
    kwargs["api_key"] = api_key
    if model_config.base_url_env:
        base_url = os.environ.get(model_config.base_url_env)
        if not base_url:
            raise ValueError(
                f"{model_config.base_url_env} environment variable is required "
                f"for model {model_config.model_id!r}. Set it in your .env file."
            )
        kwargs["base_url"] = base_url
    return ChatAnthropic(**kwargs)


def assume_role_for_task(role_arn: str, session_name: str = "patient-agent-bench") -> bool:
    """
    Assume a specific IAM role.

    This function is used to assume AWS IAM roles for:
    - Initial authentication (via assume_target_role)
    - Parallel task execution with role pool distribution

    Handles errors gracefully by logging and continuing with base credentials.

    Args:
        role_arn: The AWS IAM role ARN to assume
        session_name: Session name for the assumed role (default: "patient-agent-bench")

    Returns:
        True if role was assumed successfully, False otherwise.
        On failure, the task continues with existing credentials.
    """
    if not role_arn:
        logger.debug("No role ARN provided, using existing credentials")
        return True

    logger.debug("Assuming role: %s", role_arn)

    try:
        # Create STS client with current credentials
        sts = boto3.client('sts')

        # Assume the specified role
        response = sts.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name,
            DurationSeconds=3600  # 1 hour
        )

        # Extract credentials from response
        credentials = response['Credentials']

        # Set the assumed role credentials as environment variables
        # This makes them available to subsequent boto3 clients
        os.environ['AWS_ACCESS_KEY_ID'] = credentials['AccessKeyId']
        os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['SecretAccessKey']
        os.environ['AWS_SESSION_TOKEN'] = credentials['SessionToken']

        logger.debug("Successfully assumed role: %s", role_arn)
        return True

    except Exception as e:
        logger.warning(
            "Failed to assume role %s: %s. Continuing with base credentials.",
            role_arn,
            e,
        )
        return False


def assume_target_role() -> bool:
    """
    Assume the target IAM role specified in AWS_ARN_ROLE environment variable.

    This is Step 2 of the two-step authentication process.
    Uses the base credentials to assume a cross-account role.

    Note: Role assumption for Bedrock calls is now handled per-client via
    create_bedrock_client_with_role(). This function is kept for backward
    compatibility but is no longer called by ensure_credentials().

    Returns True if successful or if no target role is configured.
    """
    target_role_arn = os.environ.get('AWS_ARN_ROLE')
    if not target_role_arn:
        logger.debug("No AWS_ARN_ROLE configured, using base credentials")
        return True
    return assume_role_for_task(target_role_arn, session_name="patient-agent-bench")


# Threading lock to prevent concurrent credential refreshes
_credential_refresh_lock = threading.Lock()


def ensure_credentials() -> bool:
    """
    Ensure valid AWS credentials are available.

    Checks that base credentials are valid. If invalid, delegates refresh to
    refresh_credentials_hook() (a no-op in this release).

    Role assumption for Bedrock is handled per-client via
    create_bedrock_client_with_role(), so this only manages base credentials.

    Thread-safe: uses a lock to prevent concurrent credential refreshes when
    called from the retry loop across parallel tasks.

    Returns True if valid credentials are available, False otherwise.
    """
    # Check if we already have valid credentials for the right account
    if check_credentials_valid():
        logger.debug("AWS credentials are valid")
        return True

    # Acquire lock to prevent concurrent credential refreshes
    with _credential_refresh_lock:
        # Double-check after acquiring lock (another thread may have refreshed)
        if check_credentials_valid():
            logger.debug("AWS credentials refreshed by another thread")
            return True

        logger.info("Attempting credential refresh via hook...")

        if not refresh_credentials_hook():
            return False

        # Verify base credentials work and belong to the right account
        if not check_credentials_valid():
            logger.error("Credentials still invalid after refresh hook")
            return False

        logger.info("AWS credentials refreshed successfully")
        return True


# Default model name for registry lookup
DEFAULT_MODEL_NAME = "claude-sonnet-4.5-bedrock"


@dataclass
class ModelConfig:
    """
    Configuration for a single LLM model.

    Two usage patterns:

    1. Registry model (simple): Just specify a registry key
       {"model": "claude-sonnet-5-bedrock"} or {"model": "gpt-5.5-api"}
       All parameters (model_id, temperature, max_tokens, provider) come from registry.

    2. Custom model (explicit): Specify all parameters yourself
       {"model_id": "your.custom.model:0", "temperature": 0.7, "max_tokens": 4096}
       All three fields are REQUIRED - no defaults applied.
       Optionally include "provider": "openai-api" to use OpenAI API.

    You cannot mix these - either use registry OR provide full custom spec.
    """

    # For registry models
    model: Optional[str] = None

    # For custom models (all three required if using custom)
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # Provider = access CHANNEL, resolved from registry or set for custom models.
    # "openai-protocol-api" = OpenAI-compatible HTTP (ChatOpenAI);
    # "anthropic-protocol-api" = Anthropic-compatible HTTP (ChatAnthropic);
    # "bedrock" = Bedrock (ChatBedrockConverse).
    # Legacy "openai-api" is accepted and normalized to "openai-protocol-api".
    provider: Optional[str] = None

    # Model vendor (informational + channel-specific feature routing), e.g.
    # "anthropic", "openai", "mistral". Resolved from the registry.
    developer: Optional[str] = None

    # Auth mode for the openai-protocol-api channel: "api_key" | "sigv4".
    auth: Optional[str] = None

    # Mantle URL path family ("/v1" or "/openai/v1") and Responses API toggle.
    api_prefix: Optional[str] = None
    use_responses_api: bool = False

    # Extended thinking token budget (integer, 1024-128000).
    # For Anthropic models on Bedrock or the Anthropic API (older enabled-thinking shape).
    # Passed as {"type": "enabled", "budget_tokens": N}.
    # None = no explicit thinking budget.
    thinking_budget: Optional[int] = None

    # Reasoning effort level (string: "low"/"medium"/"high"/"xhigh"/"max").
    # For OpenAI reasoning models: passed as reasoning_effort to ChatOpenAI.
    # For Anthropic API (Sonnet 5 / Opus 4.7+): passed as reasoning_effort to
    # ChatAnthropic (auto-enables adaptive thinking).
    # None = use model defaults.
    reasoning_effort: Optional[str] = None

    # Additional model request fields passed to the provider API.
    # For Bedrock Converse API, these map to additionalModelRequestFields.
    # Example: {"anthropic_beta": ["context-1m-2025-08-07"]} to enable 1M context.
    additional_fields: Optional[dict] = None

    # Optional suffix appended to the system prompt to enable thinking mode.
    # For Qwen3 models: set to "/think" to enable reasoning mode.
    # None = no modification to system prompt.
    thinking_prompt_suffix: Optional[str] = None

    # Optional names of env vars holding the OpenAI-compatible base URL and
    # API key for this model. Only used when provider == "openai-api".
    # When unset, ChatOpenAI uses its defaults (api.openai.com + OPENAI_API_KEY).
    base_url_env: Optional[str] = None
    api_key_env: Optional[str] = None

    # Opt-in escape hatch for Bedrock models that DON'T support the Converse API
    # (e.g. Custom Model Import models — imported-model ARNs return a 400
    # "This action doesn't support the model" from Converse). When set, the model
    # is created via LangChain's ChatBedrock (InvokeModel under the hood) using
    # this string as the provider adapter, instead of ChatBedrockConverse.
    # The provider selects request/response serialization: "qwen" (and "openai")
    # speak the OpenAI chat-completions schema imported vLLM models return.
    # Works for any role (assistant / user / evaluator / etc.). None = default
    # Converse path (fully backward compatible).
    bedrock_invoke_provider: Optional[str] = None

    # Optional AWS region override for this specific model's Bedrock client,
    # applied on both Bedrock paths (Converse and bedrock_invoke_provider).
    # Needed when a model lives in a different region than AWS_REGION (e.g. an
    # imported model in us-east-2 while the run defaults to us-west-2). None =
    # use the run-level AWS_REGION. Ignored by the non-Bedrock channels, which
    # take their endpoint from base_url_env instead.
    region: Optional[str] = None

    def to_dict(self) -> dict:
        """
        Serialize ModelConfig back to dictionary format.

        Returns a dict with all fields: model, model_id, temperature, max_tokens, provider.
        """
        d = {
            "model": self.model,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": self.provider,
            "developer": self.developer,
            "auth": self.auth,
            "thinking_budget": self.thinking_budget,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.api_prefix:
            d["api_prefix"] = self.api_prefix
        if self.use_responses_api:
            d["use_responses_api"] = self.use_responses_api
        if self.additional_fields:
            d["additional_fields"] = self.additional_fields
        if self.thinking_prompt_suffix:
            d["thinking_prompt_suffix"] = self.thinking_prompt_suffix
        if self.base_url_env:
            d["base_url_env"] = self.base_url_env
        if self.api_key_env:
            d["api_key_env"] = self.api_key_env
        if self.bedrock_invoke_provider:
            d["bedrock_invoke_provider"] = self.bedrock_invoke_provider
        if self.region:
            d["region"] = self.region
        return d

    @property
    def is_openai_api(self) -> bool:
        """True if this model uses the OpenAI-protocol channel (not Bedrock).

        Retained for backward compatibility; equivalent to
        provider == "openai-protocol-api".
        """
        return self.provider == "openai-protocol-api"

    @property
    def requires_bedrock(self) -> bool:
        """True if this model is called through the AWS Bedrock Converse channel
        (needs a boto3 bedrock client + AWS credentials).

        False for API-key channels — the OpenAI protocol (openai-protocol-api)
        and the Anthropic protocol (anthropic-protocol-api) — which authenticate
        with an API key (or SigV4, for a gateway) rather than a bedrock client.
        """
        return self.provider == "bedrock"

    def __post_init__(self):
        """Validate and resolve configuration."""
        # Back-compat: normalize the legacy channel name.
        if self.provider == "openai-api":
            self.provider = "openai-protocol-api"

        has_model = self.model is not None
        has_custom = any([
            self.model_id is not None,
            self.temperature is not None,
            self.max_tokens is not None,
        ])

        if has_model and has_custom:
            # Check if model is a known registry name
            spec = get_model_spec(self.model)
            if spec is not None:
                raise ValueError(
                    "Cannot mix registry model with custom parameters. "
                    "Either use 'model' (registry) OR provide 'model_id', 'temperature', 'max_tokens' (custom)."
                )
            # Non-registry model name with custom params — treat as custom
            # but preserve the display name (e.g., a custom label).
            # temperature is optional: some models (e.g. Sonnet 5, Opus 4.8)
            # reject the `temperature` param, so a None value is legitimate and
            # is omitted from the request by create_chat_model().
            missing = []
            if self.model_id is None:
                missing.append("model_id")
            if self.max_tokens is None:
                missing.append("max_tokens")
            if missing:
                raise ValueError(
                    f"Custom model config missing required fields: {', '.join(missing)}. "
                    "Provide all of: model_id, max_tokens (temperature optional)."
                )
            # Keep self.model as-is (the display name)
            if self.provider is None:
                self.provider = "bedrock"

        elif has_model:
            # Registry model - resolve from registry
            spec = get_model_spec(self.model)
            if spec is None:
                raise ValueError(
                    f"Unknown model '{self.model}'. "
                    f"Use a registry key or provide a custom spec with model_id, temperature, max_tokens."
                )
            self.model_id = spec.model_id
            self.temperature = spec.default_temperature
            self.max_tokens = spec.default_max_tokens
            self.provider = spec.provider
            self.developer = spec.developer
            self.auth = spec.auth
            self.api_prefix = spec.api_prefix
            self.use_responses_api = spec.use_responses_api
            self.base_url_env = spec.base_url_env
            self.api_key_env = spec.api_key_env

        elif has_custom:
            # Custom model - model_id and max_tokens required.
            # temperature is optional: some models (e.g. Opus 4.8, Sonnet 5)
            # reject the `temperature` param, so a None value is legitimate and
            # is omitted from the request by create_chat_model().
            missing = []
            if self.model_id is None:
                missing.append("model_id")
            if self.max_tokens is None:
                missing.append("max_tokens")

            if missing:
                raise ValueError(
                    f"Custom model config missing required fields: {', '.join(missing)}. "
                    "Provide all of: model_id, max_tokens (temperature optional)."
                )
            # Mark as custom model for clarity
            self.model = "custom"
            # Provider defaults to bedrock if not explicitly set
            if self.provider is None:
                self.provider = "bedrock"
        else:
            # No config provided - use default registry model
            spec = get_model_spec(DEFAULT_MODEL_NAME)
            self.model = DEFAULT_MODEL_NAME
            self.model_id = spec.model_id
            self.temperature = spec.default_temperature
            self.max_tokens = spec.default_max_tokens
            self.provider = spec.provider
            self.developer = spec.developer
            self.auth = spec.auth
            self.api_prefix = spec.api_prefix
            self.use_responses_api = spec.use_responses_api
            self.base_url_env = spec.base_url_env
            self.api_key_env = spec.api_key_env


def _warn_if_registry_override(model_name: str, spec, d: dict) -> None:
    """Warn when a registry key is combined with temperature/max_tokens that
    differ from the registry defaults.

    By design a registry key always resolves to the spec's defaults; to
    customize temperature/max_tokens you must use a full custom spec (model_id)
    instead. to_dict() round-trips re-emit the resolved defaults, so we stay
    silent when the provided values match the spec (a round-trip) and only warn
    when a hand-authored override *differs* (i.e. the user expected it to take
    effect but it will be ignored).
    """
    differs = (
        ("temperature" in d and d["temperature"] != spec.default_temperature)
        or ("max_tokens" in d and d["max_tokens"] != spec.default_max_tokens)
    )
    if differs:
        logger.warning(
            "Model '%s' is a registry key; its temperature/max_tokens are "
            "fixed at the registry defaults (temperature=%s, max_tokens=%s) and "
            "the values you provided (temperature=%s, max_tokens=%s) are "
            "ignored. To customize these, use a full custom spec with 'model_id' "
            "instead of the '%s' registry key.",
            model_name,
            spec.default_temperature,
            spec.default_max_tokens,
            d.get("temperature"),
            d.get("max_tokens"),
            model_name,
        )


def parse_model_config(d: dict) -> ModelConfig:
    """
    Parse a single model config dict into a ModelConfig instance.

    Handles registry models, custom models, and round-trip serialized dicts.
    """
    if not d:
        return ModelConfig()  # Use default registry model

    # Check which format is being used
    has_model = "model" in d
    has_custom = "model_id" in d

    # If model has explicit custom fields (model_id, temperature, etc.),
    # treat as custom model format — but only if it's NOT a registry name
    # (registry names with extra fields come from to_dict() round-trips)
    additional_fields = d.get("additional_fields")
    thinking_prompt_suffix = d.get("thinking_prompt_suffix")

    if has_model and has_custom:
        model_name = d["model"]
        spec = get_model_spec(model_name)
        if spec is not None:
            # Registry model — ignore serialized fields, use registry defaults.
            _warn_if_registry_override(model_name, spec, d)
            return ModelConfig(model=model_name, thinking_budget=d.get("thinking_budget"), reasoning_effort=d.get("reasoning_effort"), additional_fields=additional_fields, thinking_prompt_suffix=thinking_prompt_suffix)
        kwargs = dict(
            model_id=d.get("model_id"),
            temperature=d.get("temperature"),
            max_tokens=d.get("max_tokens"),
            provider=d.get("provider"),
            developer=d.get("developer"),
            auth=d.get("auth"),
            api_prefix=d.get("api_prefix"),
            use_responses_api=d.get("use_responses_api", False),
            thinking_budget=d.get("thinking_budget"),
            reasoning_effort=d.get("reasoning_effort"),
            additional_fields=additional_fields,
            thinking_prompt_suffix=thinking_prompt_suffix,
            base_url_env=d.get("base_url_env"),
            api_key_env=d.get("api_key_env"),
            bedrock_invoke_provider=d.get("bedrock_invoke_provider"),
            region=d.get("region"),
        )
        if model_name != "custom":
            kwargs["model"] = model_name
        return ModelConfig(**kwargs)
    if has_model:
        model_name = d["model"]
        spec = get_model_spec(model_name)
        if spec is not None:
            _warn_if_registry_override(model_name, spec, d)
        return ModelConfig(model=model_name, thinking_budget=d.get("thinking_budget"), reasoning_effort=d.get("reasoning_effort"), additional_fields=additional_fields, thinking_prompt_suffix=thinking_prompt_suffix)
    if has_custom:
        return ModelConfig(
            model_id=d.get("model_id"),
            temperature=d.get("temperature"),
            max_tokens=d.get("max_tokens"),
            provider=d.get("provider"),
            developer=d.get("developer"),
            auth=d.get("auth"),
            api_prefix=d.get("api_prefix"),
            use_responses_api=d.get("use_responses_api", False),
            thinking_budget=d.get("thinking_budget"),
            reasoning_effort=d.get("reasoning_effort"),
            additional_fields=additional_fields,
            thinking_prompt_suffix=thinking_prompt_suffix,
            base_url_env=d.get("base_url_env"),
            api_key_env=d.get("api_key_env"),
            bedrock_invoke_provider=d.get("bedrock_invoke_provider"),
            region=d.get("region"),
        )
    return ModelConfig()  # Use default registry model


def parse_model_configs(value) -> List[ModelConfig]:
    """Parse model config(s), normalizing single dict to list."""
    if value is None:
        return [ModelConfig()]  # Default
    if isinstance(value, dict):
        return [parse_model_config(value)]
    if isinstance(value, list):
        return [parse_model_config(v) for v in value]
    return [ModelConfig()]  # Default


def as_list(value) -> list:
    """Normalize a value to a list: wrap a single dict in a list, pass lists through."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


@dataclass
class AgentSpec:
    """
    Configuration for a single agent variant.

    Bundles model selection, prompt selection, agent architecture,
    and optional metadata into one unit of variation.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: str = "default_prompt"
    agent_class: str = "default"
    system_prompt: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize AgentSpec to a JSON-compatible dict."""
        return {
            "model": self.model.to_dict(),
            "prompt": self.prompt,
            "agent_class": self.agent_class,
            "system_prompt": self.system_prompt,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSpec":
        """
        Parse an AgentSpec from a config dict.

        Supports two formats:
        1. Nested model key: {"model": {...}, "prompt": "...", ...}
        2. Flat ModelConfig dict: {"model_id": "...", "temperature": 0.7, ...}
           (entire dict treated as ModelConfig)
        """
        model_data = data.get("model", data)  # support nested or flat
        model = parse_model_config(model_data)
        return cls(
            model=model,
            prompt=data.get("prompt", "default_prompt"),
            agent_class=data.get("agent_class", "default"),
            system_prompt=data.get("system_prompt"),
            label=data.get("label"),
        )


@dataclass
class BenchConfig:
    """
    Benchmark configuration for PatientAgentBench.

    Loaded from a JSON config file. AWS settings (region, credentials)
    come from .env / environment variables separately.

    Supports multiple agent configurations per role for multi-experiment runs.
    JSON keys use singular form (assistant_agent, user_agent, evaluator_model).
    Prompts live inside each AgentSpec — no top-level prompt fields.
    """

    # Lists of agent specs for assistant and user roles
    assistant_agents: List[AgentSpec] = field(default_factory=lambda: [AgentSpec()])
    user_agents: List[AgentSpec] = field(default_factory=lambda: [AgentSpec()])

    # Evaluator models remain as plain ModelConfig list (no AgentSpec wrapping)
    evaluator_models: List[ModelConfig] = field(default_factory=lambda: [ModelConfig()])

    # Sandbox model config (single model, used for generating sandbox data)
    sandbox_model: ModelConfig = field(default_factory=ModelConfig)

    # Seed generator model config (single model, used for enriching benchmark seeds)
    seed_generator_model: ModelConfig = field(default_factory=ModelConfig)

    # Analyzer model config (single model, used for generating evaluation insights)
    analyzer_model: ModelConfig = field(default_factory=ModelConfig)

    # General settings
    max_turns: int = 3

    # Response processing. Defaults to True: extended-thinking blocks are
    # scratchpad reasoning, not part of the patient-facing reply, so they are
    # stripped before the response is scored.
    strip_thinking_content: bool = True

    # Aggregation method for multi-evaluator results
    aggregation_method: str = "average"

    # Backward compatibility properties for single-model access
    @property
    def assistant_model(self) -> ModelConfig:
        """Get the first assistant model (backward compatibility)."""
        return self.assistant_agents[0].model

    @property
    def user_model(self) -> ModelConfig:
        """Get the first user model (backward compatibility)."""
        return self.user_agents[0].model

    @property
    def evaluator_model(self) -> ModelConfig:
        """Get the first evaluator model (backward compatibility)."""
        return self.evaluator_models[0]

    @classmethod
    def from_file(cls, path: str) -> "BenchConfig":
        """Load configuration from a JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchConfig":
        """
        Create configuration from a dictionary.

        Parses assistant_agent and user_agent keys as lists of AgentSpec objects.
        Handles single-dict-instead-of-list wrapping via as_list().
        Defaults to [AgentSpec()] when keys are missing.
        Evaluator models remain as plain ModelConfig list.

        AgentSpec format:
        {
            "assistant_agent": [
                {
                    "model": {"model": "claude-sonnet-5-bedrock"},
                    "prompt": "default_prompt",
                    "agent_class": "default",
                    "label": "Sonnet 5"
                }
            ],
            "user_agent": {"model": {"model": "claude-haiku-4.5-bedrock"}, "prompt": "default_prompt"},
            "evaluator_model": [{"model": "claude-sonnet-5-bedrock"}, {"model": "claude-haiku-4.5-bedrock"}]
        }
        """
        # Parse assistant agents from assistant_agent key
        assistant_agents = [
            AgentSpec.from_dict(d)
            for d in as_list(data.get("assistant_agent", []))
        ] or [AgentSpec()]

        # Parse user agents from user_agent key
        user_agents = [
            AgentSpec.from_dict(d)
            for d in as_list(data.get("user_agent", []))
        ] or [AgentSpec()]

        # Parse evaluator models (plain ModelConfig, not AgentSpec)
        evaluator_models = parse_model_configs(data.get("evaluator_model"))

        kwargs: dict = {
            "assistant_agents": assistant_agents,
            "user_agents": user_agents,
            "evaluator_models": evaluator_models,
        }

        # Parse sandbox_model (single model, uses default if not specified)
        kwargs["sandbox_model"] = parse_model_config(data.get("sandbox_model", {}))

        # Parse seed_generator_model (single model, used for enriching benchmark seeds)
        kwargs["seed_generator_model"] = parse_model_config(data.get("seed_generator_model", {}))

        # Parse analyzer_model (single model, used for generating evaluation insights)
        kwargs["analyzer_model"] = parse_model_config(data.get("analyzer_model", {}))

        if "max_turns" in data:
            kwargs["max_turns"] = data["max_turns"]

        # Parse strip_thinking_content (optional; defaults to True). Set to
        # false to keep <thinking>...</thinking> content in assistant responses.
        if "strip_thinking_content" in data:
            kwargs["strip_thinking_content"] = data["strip_thinking_content"]

        # Parse aggregation_method (default to "average" if not specified)
        valid_aggregation_methods = ["average", "majority_vote"]
        aggregation_method = data.get("aggregation_method", "average")
        if aggregation_method not in valid_aggregation_methods:
            raise ValueError(
                f"Invalid aggregation_method '{aggregation_method}'. "
                f"Valid options: {valid_aggregation_methods}"
            )
        kwargs["aggregation_method"] = aggregation_method

        return cls(**kwargs)

    def to_dict(self) -> dict:
        """
        Serialize full config with AgentSpec lists for assistant/user agents.

        JSON keys use singular form (assistant_agent, user_agent, evaluator_model).
        """
        return {
            "assistant_agent": [a.to_dict() for a in self.assistant_agents],
            "user_agent": [u.to_dict() for u in self.user_agents],
            "evaluator_model": [m.to_dict() for m in self.evaluator_models],
            "sandbox_model": self.sandbox_model.to_dict(),
            "seed_generator_model": self.seed_generator_model.to_dict(),
            "analyzer_model": self.analyzer_model.to_dict(),
            "max_turns": self.max_turns,
            "strip_thinking_content": self.strip_thinking_content,
            "aggregation_method": self.aggregation_method,
        }

    @classmethod
    def default(cls) -> "BenchConfig":
        """Create default configuration."""
        return cls()
