class PromptManager:
    def __init__(self):
        # Base templates for different purposes
        self.base_templates = {
            "default": "You are a helpful AI assistant.",
            "coding": "You are a coding assistant. Provide code snippets and explanations when asked.",
            "creative": "You are a creative AI. Think outside the box and provide novel responses.",
            "concise": "You are a concise AI. Keep your answers brief and to the point.",
            "scientific": "You are a scientific AI assistant. Provide accurate information with references when possible."
        }
        
        # Model-specific template formats
        self.model_formats = {
            "llama": {
                "system": "<|system|>\n{content}\n",
                "user": "<|user|>\n{content}\n",
                "assistant": "<|assistant|>\n{content}\n"
            },
            "mistral": {
                "system": "<s>[INST] {content} [/INST]",
                "user": "[INST] {content} [/INST]",
                "assistant": "{content}"
            },
            "qwen": {
                "system": "<|im_start|>system\n{content}<|im_end|>\n",
                "user": "<|im_start|>user\n{content}<|im_end|>\n",
                "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n"
            },
            "deepseek": {
                "system": "<｜system｜>\n{content}\n",
                "user": "<｜user｜>\n{content}\n",
                "assistant": "<｜assistant｜>\n{content}\n"
            }
        }
        
    def get_template(self, name):
        """Get a template by name"""
        return self.base_templates.get(name, self.base_templates["default"])
        
    def add_template(self, name, template):
        """Add a new template"""
        self.base_templates[name] = template
        
    def list_templates(self):
        """List all available templates"""
        return list(self.base_templates.keys())
        
    def format_system_message(self, template_name, model_family="generic", **kwargs):
        """Format a system message for a specific model family"""
        template_content = self.get_template(template_name)
        
        # If no specific format exists for this model family, return the raw template
        if model_family not in self.model_formats:
            return template_content
            
        # Format the template according to the model's system message format
        system_format = self.model_formats[model_family]["system"]
        return system_format.format(content=template_content.format(**kwargs))
    
    def get_format_for_model(self, model_family):
        """Get the message format templates for a specific model family"""
        return self.model_formats.get(model_family, self.model_formats.get("llama"))