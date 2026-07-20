from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate  # prompt template
from pydantic import BaseModel # for the json or structure 
from typing import List, Optional # oops and type 
from langchain_core.output_parsers import PydanticOutputParser # parser


class Ingredient(BaseModel):
    name: str
    quantity: str


class Recipe(BaseModel):
    dish: str
    description: str
    preparation_time: str
    cooking_time: str
    total_time: str
    servings: int
    ingredients: List[Ingredient]
    instructions: List[str]
    tips: List[str]
    nutrition: Optional[List[str]]


parser = PydanticOutputParser(pydantic_object=Recipe)


model = ChatMistralAI(model="mistral-small-2506")

# promptForNormal = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
# You are an expert chef.

# When the user provides the name of a dish, generate a complete recipe in plain text.

# Your response should include:
# - A short introduction about the dish.
# - Preparation time.
# - Cooking time.
# - Servings.
# - Ingredients (as a simple bullet list).
# - Step-by-step cooking instructions.
# - Optional tips for making the dish better.

# Do NOT return JSON, XML, Markdown tables, or any structured format.
# Return a natural, human-readable recipe only.
#             """,
#         ),
#         (
#             "human",
#             "Give me the recipe for {dish_name}.",
#         ),
#     ]
# )


promptForJson = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert chef.

When the user provides the name of a dish, generate a complete recipe.

The recipe must include:
- dish
- description
- preparation_time
- cooking_time
- total_time
- servings
- ingredients (each with name and quantity)
- instructions (step-by-step)
- tips
- nutrition (if available)

{format_instructions}

Do not include any explanation, markdown, or extra text.
Return only the JSON object.
            """,
        ),
        (
            "human",
            "Generate the recipe for: {dish_name}",
        ),
    ]
)

dish_name = input("Give the dish you want to make : ")

final_prompt = promptForJson.invoke(
    {"dish_name": dish_name, "format_instructions": parser.get_format_instructions()}
)


response = model.invoke(final_prompt)
recipe = parser.parse(response.content)
print(recipe)
