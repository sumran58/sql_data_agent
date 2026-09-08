import os
import sys

# Add project root to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from models.schema import EtlAgentSchema


# ============================================================
# 1. EXTRACT + LOAD TOOL
# ============================================================

@tool
def extract_load_tool(
    url: str,
    output_folder: str,
    format: str
) -> str:
    """
    Extract data from an API and save it to the requested folder.

    Args:
        url: API endpoint.
        output_folder: Folder where extracted data should be saved.
        format: Output format such as csv, json, or parquet.

    Returns:
        Short status message describing the operation.
    """

    etl_tools = ETLTools()

    result = etl_tools.extract_load(
        url,
        output_folder,
        format
    )

    # Keep the tool response short.
    # Do not send unnecessarily large output back to the LLM.
    return str(result)


# ============================================================
# 2. TRANSFORM + LOAD TOOL
# ============================================================

@tool
def transform_load_tool(
    input_file_path: str,
    output_folder: str,
    output_format: str,
    user_question: str
) -> str:
    """
    Transform data from an input file according to the user's
    question and save the transformed data.

    Args:
        input_file_path: Path to input file.
        output_folder: Destination folder.
        output_format: Output format such as csv, json, or parquet.
        user_question: User's requested transformation.

    Returns:
        Short status message.
    """

    etl_tools = ETLTools()

    # --------------------------------------------------------
    # Get a small sample of the data.
    # --------------------------------------------------------

    top_3_rows = etl_tools.transform_load_context(
        input_file_path
    )

    # --------------------------------------------------------
    # Use the medium LLM for generating Pandas code.
    # --------------------------------------------------------

    llm = pick_llm("medium")

    prompt = f"""
You are a Python Data Analyst specializing in Pandas.

Generate ONLY executable Pandas Python code.

Task:
Transform the data stored in:
{input_file_path}

Save the transformed data to:
{output_folder}

Output format:
{output_format}

User request:
{user_question}

Sample data:
{top_3_rows}

Requirements:
- Load the input file into a Pandas DataFrame.
- Perform the transformation requested by the user.
- Save the resulting DataFrame to the requested output location.
- Use Pandas.
- Return ONLY Python code.
- Do not use markdown.
- Do not use ```python.
- Do not provide explanations.
"""

    response = llm.invoke(prompt)

    pandas_code = response.content.strip()

    # --------------------------------------------------------
    # Clean markdown code fences if the model accidentally
    # returns them.
    # --------------------------------------------------------

    if pandas_code.startswith("```python"):
        pandas_code = pandas_code[len("```python"):].strip()

    elif pandas_code.startswith("```"):
        pandas_code = pandas_code[len("```"):].strip()

    if pandas_code.endswith("```"):
        pandas_code = pandas_code[:-3].strip()

    # --------------------------------------------------------
    # Execute generated Pandas code.
    # --------------------------------------------------------

    print("\n========== GENERATED PANDAS CODE ==========\n")
    print(pandas_code)

    results = etl_tools.execute_code(
        pandas_code
    )

    print("\n========== EXECUTION RESULT ==========\n")
    print(results)

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT return the generated code + full execution result
    # to the main agent.
    #
    # This prevents the LangGraph message history from becoming
    # unnecessarily large.
    # --------------------------------------------------------

    return (
        f"Transformation completed successfully. "
        f"The transformed data was saved to "
        f"'{output_folder}' in {output_format} format."
    )


# ============================================================
# 3. TOOLS
# ============================================================

tools = [
    extract_load_tool,
    transform_load_tool
]


# ============================================================
# 4. LLM
# ============================================================

# Keep your current model.
# The problem was the request size, not necessarily the model.

llm = pick_llm("medium")

llm_bind = llm.bind_tools(tools)


# ============================================================
# 5. LLM NODE
# ============================================================

def llm_node(state: EtlAgentSchema):
    """
    Main reasoning node.

    The LLM decides whether an ETL tool needs to be called.
    """

    messages = state.messages

    system_prompt = """
You are an ETL Analyst Agent.

You have access to these tools:

1. extract_load_tool
   - Extracts data from an API.
   - Saves the data to a specified location.

2. transform_load_tool
   - Transforms an existing data file.
   - Saves the transformed data.

Your job:
- Understand the user's request.
- Select the appropriate tool.
- Call the tool with the correct arguments.
- After the tool completes successfully, provide a concise final response.
- Do not call a tool unnecessarily.
- Do not repeat an already completed operation.
"""

    # --------------------------------------------------------
    # IMPORTANT:
    # We pass the actual LangChain messages instead of
    # converting the entire message list into a huge string.
    # --------------------------------------------------------

    prompt_messages = [
        HumanMessage(content=system_prompt)
    ] + messages

    final_answer = llm_bind.invoke(
        prompt_messages
    )

    # Add the LLM response to the existing conversation.
    state.messages = messages + [final_answer]

    return state


# ============================================================
# 6. TOOL NODE
# ============================================================

def tool_node(state: EtlAgentSchema):
    """
    Executes tools requested by the LLM.
    """

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    last_message = state.messages[-1]

    tool_calls = getattr(
        last_message,
        "tool_calls",
        []
    )

    tools_results = []

    for tool_call in tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]

        selected_tool = tools_by_name.get(
            tool_name
        )

        if selected_tool is None:

            tools_results.append(
                ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_call["id"]
                )
            )

            continue

        try:

            observation = selected_tool.invoke(
                tool_args
            )

            # Keep observation as a string.
            observation = str(observation)

        except Exception as e:

            observation = (
                f"Tool execution failed: {str(e)}"
            )

        tools_results.append(
            ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"]
            )
        )

    # Add tool results to the state.
    state.messages = (
        state.messages + tools_results
    )

    return state


# ============================================================
# 7. ROUTING FUNCTION
# ============================================================

def is_tool_call(state: EtlAgentSchema):
    """
    Decide whether the graph should execute a tool
    or finish the conversation.
    """

    last_message = state.messages[-1]

    tool_calls = getattr(
        last_message,
        "tool_calls",
        []
    )

    if tool_calls:
        return "tool_node"

    return "end"


# ============================================================
# 8. BUILD LANGGRAPH
# ============================================================

etl_analyst_graph = StateGraph(
    EtlAgentSchema
)

# Add nodes
etl_analyst_graph.add_node(
    "llm_node",
    llm_node
)

etl_analyst_graph.add_node(
    "tool_node",
    tool_node
)

# START → LLM
etl_analyst_graph.add_edge(
    START,
    "llm_node"
)

# LLM → Tool OR END
etl_analyst_graph.add_conditional_edges(
    "llm_node",
    is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END
    }
)

# Tool → LLM
etl_analyst_graph.add_edge(
    "tool_node",
    "llm_node"
)


# Compile
etl_analyst = etl_analyst_graph.compile()


# ============================================================
# 9. TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Generate graph image
    # --------------------------------------------------------

    from IPython.display import Image

    img = Image(
        etl_analyst
        .get_graph()
        .draw_mermaid_png()
    )

    with open(
        "etl_analyst_graph.png",
        "wb"
    ) as f:
        f.write(img.data)

    # --------------------------------------------------------
    # TEST TRANSFORMATION
    # --------------------------------------------------------

    response = etl_analyst.invoke(
        {
            "messages": [
                HumanMessage(
                    content="""
I want to transform the data stored in
'data/extract/extracted data.csv'
and save the transformed data in
'data/extract/transform'
in CSV format.

The transformation should filter the data
to show Bulbasaur Pokemon only.
"""
                )
            ]
        }
    )

    # --------------------------------------------------------
    # Print final response
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL RESPONSE")
    print("========================================\n")

    print(response)