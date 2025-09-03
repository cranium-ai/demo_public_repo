from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

# Explicitly create a response for each Google Gemini model
response_gemini_1_5_flash = client.models.generate_content(
    model="gemini-1.5-flash", contents="Explain how AI works in a few words"
)
response_gemini_1_5_flash_8b = client.models.generate_content(
    model="gemini-1.5-flash-8b", contents="Explain how AI works in a few words"
)
response_gemini_1_5_pro = client.models.generate_content(
    model="gemini-1.5-pro", contents="Explain how AI works in a few words"
)
response_gemini_2_0_flash = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
response_gemini_2_0_flash_lite = client.models.generate_content(
    model="gemini-2.0-flash-lite", contents="Explain how AI works in a few words"
)
response_gemini_2_5_flash = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how AI works in a few words"
)
response_gemini_2_5_flash_lite_preview_06_17 = client.models.generate_content(
    model="gemini-2.5-flash-lite-preview-06-17",
    contents="Explain how AI works in a few words",
)
response_gemini_2_5_pro = client.models.generate_content(
    model="gemini-2.5-pro", contents="Explain how AI works in a few words"
)
