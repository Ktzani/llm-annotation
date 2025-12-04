# =============================================================================
# PROMPT TEMPLATES - Otimizados para anotação de dados
# =============================================================================

# BASE_ANNOTATION_PROMPT = """You are an expert data annotator with extensive experience in text classification tasks.

# Your task is to classify the following text into one of the predefined categories with high precision.

# **Instructions:**
# 1. Read the text carefully and understand its context
# 2. Consider the nuances and implicit meanings
# 3. Select the most appropriate category based on the content
# 4. Be consistent with your classification criteria
# 5. If the text is ambiguous, choose the most likely category based on dominant features

# **Available Categories:**
# {categories}

# **Text to classify ({description}):**
# {text}

# **Important Guidelines:**
# - Provide ONLY the category number as your response
# - Do not include explanations
# - Be objective and avoid bias
# - Consider edge cases carefully
# - Maintain consistency across similar texts

# **Your classification to the {description_lower} provided (category number only OR I WILL DIE):**"""

# # Prompt base com técnicas de prompt engineering
BASE_ANNOTATION_PROMPT = """You are an expert data annotator with extensive experience in text classification tasks.

Your task is to classify the following text into one of the predefined categories with high precision.

**Instructions:**
1. Read the text carefully and understand its context
2. Consider the nuances and implicit meanings
3. Select the most appropriate category based on the content
4. Be consistent with your classification criteria
5. If the text is ambiguous, choose the most likely category based on dominant features

**Available Categories:**
{categories}

**Text to classify ({description}):**
{text}

**Important Guidelines:**
- Provide ONLY the category number as your response
- Do not include explanations
- Be objective and avoid bias
- Consider edge cases carefully
- Maintain consistency across similar texts

**Your classification to the {description_lower} provided (category number only):**"""

# Prompt com few-shot learning (adicionar exemplos quando disponível)
FEW_SHOT_PROMPT = """You are an expert data annotator. Below are examples of correctly classified texts:

{examples}

Now, classify the following text using the same criteria:

**Text to classify ({description}):**
{text}

**Available Categories:**
{categories}

**Your classification to the {description_lower} provided (category number only):**"""

# Prompt com Chain-of-Thought para casos complexos
COT_PROMPT = """You are an expert data annotator. Analyze the following text step by step:

**Text to classify ({description}):**
{text}

**Available Categories:**
{categories}

**Analysis Process:**
1. First, identify the main topic or theme
2. Consider the tone and intent
3. Look for key indicators that match each category
4. Evaluate which category best fits

After your analysis, provide ONLY the final category name on the last line preceded by "CLASSIFICATION:"

**Your response to the {description_lower} provided (category number only):**"""


SIMPLER_PROMPT = """{description}: {text}. Based on the content of the {description_lower} provided, which of the
following categories would it best fit under: {categories}?
Just select one of these options. No explanation required. Just the category number."""