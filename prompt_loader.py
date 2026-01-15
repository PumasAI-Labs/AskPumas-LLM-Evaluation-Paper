from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def load_prompt(template_name: str, templates_dir: str = "prompts/templates") -> str:
    """
    Load a prompt from a Jinja2 template file.
    
    Args:
        template_name: Name of the template file (without .j2 extension)
        templates_dir: Directory containing template files
        
    Returns:
        Rendered prompt string
    """
    base_dir = Path(__file__).parent
    template_path = base_dir / templates_dir
    
    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template(f"{template_name}.j2")
    
    return template.render()
