"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl
from typing import List, Optional, Sequence, Union

from vllm.engine.arg_utils import AsyncEngineArgs, nullable_str
from vllm.entrypoints.chat_utils import validate_chat_template
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.utils import FlexibleArgumentParser




def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    parser.add_argument("--host",
                        type=nullable_str,
                        default=None,
                        help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=['debug', 'info', 'warning', 'error', 'critical', 'trace'],
        help="log level for uvicorn")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--api-key",
                        type=nullable_str,
                        default=None,
                        help="If provided, the server will require this key "
                        "to be presented in the header.")
    parser.add_argument("--chat-template",
                        type=nullable_str,
                        default=None,
                        help="The file path to the chat template, "
                        "or the template in single-line form "
                        "for the specified model")
    parser.add_argument("--response-role",
                        type=nullable_str,
                        default="assistant",
                        help="The role name to return if "
                        "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=nullable_str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=nullable_str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument("--ssl-ca-certs",
                        type=nullable_str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=nullable_str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=nullable_str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
        "We accept multiple --middleware arguments. "
        "The value should be an import path. "
        "If a function is provided, vLLM will add it to the server "
        "using @app.middleware('http'). "
        "If a class is provided, vLLM will add it to the server "
        "using app.add_middleware(). ")
    parser.add_argument(
        "--return-tokens-as-token-ids",
        action="store_true",
        help="When --max-logprobs is specified, represents single tokens as "
        "strings of the form 'token_id:{token_id}' so that tokens that "
        "are not JSON-encodable can be identified.")
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        help="If specified, will run the OpenAI frontend server in the same "
        "process as the model serving engine.")

    parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        default=False,
        help=
        "Enable auto tool choice for supported models. Use --tool-call-parser"
        "to specify which parser to use")

    valid_tool_parsers = ToolParserManager.tool_parsers.keys()
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        metavar="{" + ",".join(valid_tool_parsers) + "} or name registered in "
        "--tool-parser-plugin",
        default=None,
        help=
        "Select the tool call parser depending on the model that you're using."
        " This is used to parse the model-generated tool call into OpenAI API "
        "format. Required for --enable-auto-tool-choice.")

    parser.add_argument(
        "--tool-parser-plugin",
        type=str,
        default="",
        help=
        "Special the tool parser plugin write to parse the model-generated tool"
        " into OpenAI API format, the name register in this plugin can be used "
        "in --tool-call-parser.")

    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument('--max-log-len',
                        type=int,
                        default=None,
                        help='Max number of prompt characters or prompt '
                        'ID numbers being printed in log.'
                        '\n\nDefault: Unlimited')

    parser.add_argument(
        "--disable-fastapi-docs",
        action='store_true',
        default=False,
        help="Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint"
    )

    return parser


def validate_parsed_serve_args(args: argparse.Namespace):
    """Quick checks for model serve args that raise prior to loading."""
    if hasattr(args, "subparser") and args.subparser != "serve":
        return

    # Ensure that the chat template is valid; raises if it likely isn't
    validate_chat_template(args.chat_template)

    # Enable auto tool needs a tool call parser to be valid
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        raise TypeError("Error: --enable-auto-tool-choice requires "
                        "--tool-call-parser")


def create_parser_for_docs() -> FlexibleArgumentParser:
    parser_for_docs = FlexibleArgumentParser(
        prog="-m vllm.entrypoints.openai.api_server")
    return make_arg_parser(parser_for_docs)
