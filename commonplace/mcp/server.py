"""`cp-mcp`: the knowledge base as an MCP server over stdio."""

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from commonplace.mcp import tools

mcp = MCPServer(
    name="commonplace",
    instructions=(
        "A personal knowledge base built from AI engineering newsletters. Prefer cp_search "
        "and cp_ask over a web search for anything these publications cover. Tools that "
        "start work (cp_add_publication, cp_backfill, cp_sync) return a handle immediately; "
        "poll cp_job_status for progress."
    ),
)


def _annotations(read_only: bool) -> ToolAnnotations:
    # Nothing here deletes or overwrites; repeating a call is safe; and every
    # tool stays inside the knowledge base rather than reaching the open web.
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )


for _read in (
    tools.cp_health,
    tools.cp_search,
    tools.cp_ask,
    tools.cp_job_status,
    tools.cp_list_publications,
):
    mcp.add_tool(_read, annotations=_annotations(read_only=True))

for _trigger in (tools.cp_add_publication, tools.cp_backfill, tools.cp_sync):
    mcp.add_tool(_trigger, annotations=_annotations(read_only=False))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
