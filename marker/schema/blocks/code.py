import html

from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Code(Block):
    block_type: BlockTypes = BlockTypes.Code
    code: str | None = None
    block_description: str = "A programming code block."

    def assemble_html(self, document, child_blocks, parent_structure):
        code = self.code or ""
        return (f"<pre>"
                f"{html.escape(code)}"
                f"</pre>")
