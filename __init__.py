from comfy.comfy_types.node_typing import IO
from comfy_api.latest import ComfyExtension, io


PromptVariables = io.Custom("MIN_PROMPT_VARIABLES")
Anything = io.Custom(IO.ANY)


class PromptVariableDictionary(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinPromptVariableDictionary",
            display_name="Prompt Variable Dictionary",
            category="Min_node/Prompt",
            description="Defines reusable prompt variables with one name = value pair per line.",
            inputs=[
                io.String.Input(
                    "variables",
                    display_name="variables (name = value)",
                    multiline=True,
                    default=(
                        "character = a woman in a red dress\n"
                        "location = a neon-lit night city\n"
                        "style = cinematic lighting"
                    ),
                ),
            ],
            outputs=[PromptVariables.Output(display_name="prompt variables")],
        )

    @classmethod
    def execute(cls, variables: str) -> io.NodeOutput:
        parsed = {}
        for line_number, raw_line in enumerate(variables.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Line {line_number} must use the format: name = value")

            name, value = line.split("=", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"Line {line_number} has an empty variable name.")
            if "{" in name or "}" in name:
                raise ValueError(f"Line {line_number} variable name cannot contain braces.")
            if name in parsed:
                raise ValueError(f"Variable '{name}' is defined more than once.")
            parsed[name] = value.strip()

        return io.NodeOutput(parsed)


class PromptTemplateRenderer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinPromptTemplateRenderer",
            display_name="Prompt Template Renderer",
            category="Min_node/Prompt",
            description="Replaces named placeholders such as {character} with values from Prompt Variable Dictionary.",
            inputs=[
                PromptVariables.Input("prompt_variables"),
                io.String.Input(
                    "template",
                    display_name="prompt template",
                    multiline=True,
                    default="portrait of {character}, standing in {location}, {style}",
                ),
            ],
            outputs=[io.String.Output(display_name="rendered prompt")],
        )

    @classmethod
    def execute(cls, prompt_variables: dict[str, str], template: str) -> io.NodeOutput:
        rendered = template
        for name, value in prompt_variables.items():
            rendered = rendered.replace("{" + name + "}", value)
        return io.NodeOutput(rendered)


class BroadcastByInputName(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MinBroadcastByInputName",
            display_name="Broadcast by Input Name",
            category="Min_node/Routing",
            description="Broadcasts a connected value to every empty, type-compatible input with the exact target name. Requires cg-use-everywhere.",
            inputs=[
                Anything.Input("source"),
                io.String.Input(
                    "target_input_name",
                    display_name="target input name",
                    default="prompt_variables",
                ),
            ],
            outputs=[Anything.Output("broadcast")],
        )

    @classmethod
    def execute(cls, source, target_input_name: str) -> io.NodeOutput:
        return io.NodeOutput(source)


class MinNodeExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [PromptVariableDictionary, PromptTemplateRenderer, BroadcastByInputName]


async def comfy_entrypoint() -> MinNodeExtension:
    return MinNodeExtension()


WEB_DIRECTORY = "./js"
__all__ = ["WEB_DIRECTORY"]
