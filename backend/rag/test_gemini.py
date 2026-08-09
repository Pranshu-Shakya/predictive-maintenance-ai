from google import genai


client = genai.Client()


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain centrifugal pump cavitation in two sentences."
)


print(response.text)