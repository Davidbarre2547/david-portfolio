import io
import os
import re
import base64

from datetime import datetime
from google import genai
from google.genai import types
from IPython.display import clear_output, display, HTML, Markdown
from PIL import Image


def _load_api_key() -> str:
    """Load the Gemini API key from Colab secrets or the environment."""
    # Prefer Colab's secret store when running in Colab.
    try:
        from google.colab import userdata
        key = userdata.get("DAVID_GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    # Fall back to environment variables (works locally / in CI).
    key = os.getenv("DAVID_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Gemini API key not found. Set the DAVID_GEMINI_API_KEY "
            "Colab secret or the GEMINI_API_KEY environment variable."
        )
    return key


GEMINI_API_KEY = _load_api_key()

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

# Model identifiers
CHAT_MODEL = "gemini-3.1-pro-preview"
IMAGE_MODEL = "gemini-3.1-flash-image-preview"

print(f"✅Client initialized...")
print(f"Chat Model: {CHAT_MODEL}")
print(f"Image Model: {IMAGE_MODEL}")

class GeminiChatbot:
  IMAGE_TRIGGERS = [
    "generate an Image", "create an Image", "draw", "make a picture",
    "generate a picture", "create a picture", "make an Image",
    "show me an image", "generate image", "create image",
    "paint", "illustrate", "sketch", "design an image",
    "visualize", "render an image", "make art", "create art",
    "/image"
  ]

  def __init__(self, client:genai.Client, chat_model:str=CHAT_MODEL,image_model:str=IMAGE_MODEL, system_prompt:str|None=None, max_history:int=50) -> None:
    self.client = client
    self.chat_model = chat_model
    self.image_model = image_model
    self.max_history = max_history
    self.system_prompt = system_prompt or (
      "You are a helpful, creative and friendly AI assistant."
      "You remember the full conversation and use that context to"
      "give relevant, coherent answers. Be concise but thorough."
    )
    # Conversation history (list of dicts with rol + parts)
    self.history: list[dict] = []
    # Image log
    self.generated_images: list[Image.Image] = []

  # ----Intent Detection----
  def _is_image_request(self, text:str)->bool:
    """Check if the user message is an image-generation request"""
    lower = text.lower().strip()
    return any(trigger in lower for trigger in self.IMAGE_TRIGGERS)

  # ----History Helpers----
  def _trim_history(self):
    if len(self.history) > self.max_history:
      self.history = self.history[-self.max_history:]

  def _add_to_history(self, role:str, text:str):
    self.history.append({"role":role, "parts":[{"text":text}]})
    self._trim_history()

  def get_history_summary(self)->str:
    lines = []
    for msg in self.history:
      role = "😼 You" if msg["role"] == "user" else "🤖 Gemini"
      text = msg["parts"][0]["text"][:120]
      lines.append(f"{role}:{text}")
    return "\n".join(lines) if lines else '(No history yet)'

  # ----Text Chat----
  def _chat(self, user_message:str)->str:
    self._add_to_history("user", user_message)

    response = self.client.models.generate_content(
      model=self.chat_model,
      contents=self.history,
      config= types.GenerateContentConfig(
        system_instruction=self.system_prompt,
        temperature=0.8,
        max_output_tokens=2048
      )
    )

    reply = response.text
    self._add_to_history("model", reply)
    return reply

  def _generate_image(self, prompt:str)->tuple[str, Image.Image|None]:
    # Build a rich prompt using conversation context
    context_hint = ""
    if self.history:
      recent = [m['parts'][0]['text'] for m in self.history[-6:]]
      context_hint = (
        "Conversation context for reference:\n" + "\n".join(recent) + "\n\n"
      )

    full_prompt = (
      f"{context_hint}"
      f"Generate an image for the following request: {prompt}"
    )

    self._add_to_history("user", prompt)

    try:
      response = self.client.models.generate_content(
          model=self.image_model,
          contents=full_prompt,
          config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            temperature=1.0,
            max_output_tokens=4096
          ))
      text_reply = ""
      image_out = None

      for part in response.candidates[0].content.parts:
        if part.text:
          text_reply += part.text
        elif part.inline_data:
          image_bytes = part.inline_data.data
          image_out = Image.open(io.BytesIO(image_bytes))
          self.generated_images.append(image_out)

      if not text_reply:
        text_reply = "Here's the generated image!"

      self._add_to_history("model", text_reply + '[image generated]')
      return text_reply, image_out
    except Exception as e:
      error_msg = f"Image generation failed: {e}"
      self._add_to_history("model", error_msg)
      return error_msg, None

  # ---- Main Entry Point ----
  def send(self, message:str)->dict:
    if self._is_image_request(message):
      text, image = self._generate_image(message)
      return {"type": "image", "text":text, "image":image}
    else:
      reply = self._chat(message)
      return {"type": "text", "text":reply, "image":None}

  def reset(self):
    self.history.clear()
    self.generated_images.clear()
    print("🗑️ Conversation history cleared.")


def render_response(result:dict):
  display(Markdown(f"**🤖 Gemini:** {result['text']}"))
  if result["image"]:
    display(result["image"])

def chat_banner():
  display(HTML("""
  
    💬 Gemini Chatbot
    
        Text chat • Image generation • Context-aware
        Type /image <prompt> to force image generation •
        Type quit to exit •
        Type history to view context •
        Type reset to clear memory
    
  
  """))
     
bot = GeminiChatbot(client)

# Multi-turn conversation demonstrating context awarness
messages = [
    "Hi! My name is David and I love building AI projects",
    "What are 3 cool project ideas I could try next?",
    "Tell me more about the second idea. What tech stack would you recommend"
]

for msg in messages:
  display(Markdown(f"**😾 You:** {msg}"))
  result = bot.send(msg)
  render_response(result)
  display(Markdown("---"))

result = bot.send("/image Generate an Image of Persian Cat")
     

render_response(result)


import gradio as gr
import tempfile

# Fresh bot instance for the Gradio UI
gradio_bot = GeminiChatbot(client)

def respond(message:str, chat_history:list):
  result = gradio_bot.send(message)

  bot_reply = result["text"]
  image_path = None

  if result["image"]:
    # Save to temp file for Gradio display
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    result["image"].save(tmp.name)
    image_path = tmp.name
    bot_reply += f"\n\n![Generated Image]({image_path})"

  chat_history.append((message, bot_reply))
  return "", chat_history, image_path


def clear_chat():
  gradio_bot.reset()
  return [], None

with gr.Blocks(title="Gemini Chatbot", theme=gr.themes.Ocean()) as demo:
  gr.Markdown("# 🤖 Gemini Context-Aware Chatbot")
  gr.Markdown(
      "Chat naturally or use **`/image `** to generate images."
      "The bot remembers your full conversation."
  )

  with gr.Row():
    with gr.Column(scale=3):
      chatbot = gr.Chatbot(height=480, label="Conversation")
      msg = gr.Textbox(placeholder="Type a message... (use /image to generate images)", label="Your Message", lines=1)
      with gr.Row():
        send_btn = gr.Button("Send 🚀", variant="primary")
        clear_btn = gr.Button("Clear 🗑️")

    with gr.Column(scale=1):
      image_output = gr.Image(label="Latest Generated Image", height=300)

  # Wire Events
  send_btn.click(respond, [msg, chatbot], [msg, chatbot, image_output])
  msg.submit(respond, [msg, chatbot], [msg, chatbot, image_output])
  clear_btn.click(clear_chat, outputs=[chatbot, image_output])

demo.launch(debug=True, share=True)

import os

save_dir = "/content/drive/MyDrive/Generative-AI-Sessions/David-GenAI/Gemini_Chatbot/Generate_Images"

os.makedirs(save_dir, exist_ok=True)

for index, image in enumerate(gradio_bot.generated_images):
  path = os.path.join(save_dir, f"image_{index+1}.png")
  image.save(path)
  print(f"💾 Saved: {path}")

if not gradio_bot.generated_images:
  print("No images generated yet!")