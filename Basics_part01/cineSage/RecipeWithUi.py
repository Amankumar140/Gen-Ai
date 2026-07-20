import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


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


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="AI Recipe Generator", page_icon="🍳", layout="centered")

st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }
        .recipe-card {
            background-color: #1a1d24;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #2a2e37;
            margin-bottom: 1rem;
        }
        .meta-badge {
            display: inline-block;
            background-color: #262a33;
            padding: 0.35rem 0.9rem;
            border-radius: 20px;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍳 AI Recipe Generator")
st.caption("Powered by Mistral AI via LangChain")

dish_name = st.text_input("What dish do you want to make?", placeholder="e.g. Butter Chicken")
generate_clicked = st.button("Generate Recipe", type="primary", use_container_width=True)

if generate_clicked:
    if not dish_name.strip():
        st.warning("Please enter a dish name first.")
    else:
        with st.spinner(f"Cooking up the recipe for {dish_name}..."):
            final_prompt = promptForJson.invoke(
                {
                    "dish_name": dish_name,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            response = model.invoke(final_prompt)

            try:
                recipe: Recipe = parser.parse(response.content)
            except Exception as e:
                st.error("Couldn't parse the model's response into structured JSON.")
                st.text_area("Raw response", response.content, height=300)
                st.stop()

        # ---- Display ----
        st.markdown(f"## {recipe.dish}")
        st.write(recipe.description)

        st.markdown(
            f"""
            <span class="meta-badge">⏱️ Prep: {recipe.preparation_time}</span>
            <span class="meta-badge">🔥 Cook: {recipe.cooking_time}</span>
            <span class="meta-badge">⌛ Total: {recipe.total_time}</span>
            <span class="meta-badge">🍽️ Servings: {recipe.servings}</span>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ingredients")
            for ing in recipe.ingredients:
                st.markdown(f"- **{ing.quantity}** {ing.name}")

        with col2:
            st.subheader("Instructions")
            for i, step in enumerate(recipe.instructions, start=1):
                st.markdown(f"**{i}.** {step}")

        if recipe.tips:
            st.subheader("💡 Tips")
            for tip in recipe.tips:
                st.markdown(f"- {tip}")

        if recipe.nutrition:
            st.subheader("Nutrition")
            for n in recipe.nutrition:
                st.markdown(f"- {n}")
        with st.expander("🔍 View Raw JSON Data"):
            st.json(recipe.model_dump())