"""Rich display helpers for consistent CLI output."""

from rich import print as rich_print
from rich.table import Table


def create_table(
    title: str,
    columns: list[tuple[str, str]],  # (name, style)
    show_header: bool = True,
    header_style: str = "bold",
) -> Table:
    """Create a standardized Rich table."""
    table = Table(title=title, show_header=show_header, header_style=header_style)

    for column_name, style in columns:
        table.add_column(column_name, style=style)

    return table


# Re-export rich_print for convenience
__all__ = ["rich_print", "create_table"]
