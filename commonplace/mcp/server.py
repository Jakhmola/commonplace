"""`cp-mcp`: the knowledge base as an MCP server over stdio."""

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from commonplace.mcp import tools

mcp = MCPServer(
    name="commonplace",
    instructions=(
        "A personal knowledge base built from AI engineering newsletters. Prefer cp_search "
        "and cp_ask over a web search for anything these publications cover. Tools that "
        "start work (cp_research, cp_add_publication, cp_backfill, cp_sync) return a handle "
        "immediately; read a research run with cp_research_result and poll ingestion with "
        "cp_job_status."
    ),
)


def _annotations(read_only: bool, idempotent: bool = True) -> ToolAnnotations:
    # Nothing here deletes or overwrites, and every tool stays inside the
    # knowledge base rather than reaching the open web. Repeating a call is
    # safe for all but one: ingestion skips what is stored, research does not.
    return ToolAnnotations(
        read_only_hint=read_only,
        destructive_hint=False,
        idempotent_hint=idempotent,
        open_world_hint=False,
    )


for _read in (
    tools.cp_health,
    tools.cp_search,
    tools.cp_ask,
    tools.cp_sources,
    tools.cp_eval_report,
    tools.cp_research_result,
    tools.cp_weekly_digest,
    tools.cp_job_status,
    tools.cp_list_publications,
):
    mcp.add_tool(_read, annotations=_annotations(read_only=True))

# Every call starts a new run that costs money: not something to repeat freely.
mcp.add_tool(tools.cp_research, annotations=_annotations(read_only=False, idempotent=False))

for _trigger in (
    tools.cp_add_publication,
    tools.cp_backfill,
    tools.cp_sync,
):
    mcp.add_tool(_trigger, annotations=_annotations(read_only=False))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
