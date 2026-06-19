const projects = [
  {
    title: "Emotion Detection",
    kicker: "Computer vision",
    description:
      "Upload a photo and it detects every face and labels the strongest emotion, drawn live on the image — runs entirely in your browser.",
    tags: ["Faces", "In-browser", "Real-time"],
    demo: "emotion.html",
  },
  {
    title: "Face Filter",
    kicker: "Live webcam effect",
    description:
      "Turns on your webcam, detects your face in real time, and places a sunglasses overlay across your eyes — all on your device.",
    tags: ["Webcam", "Face landmarks", "Live"],
    demo: "filter.html",
  },
  {
    title: "Gemini ChatBot",
    kicker: "Generative AI",
    description:
      "A multimodal chatbot with text conversation, image generation, and conversation memory — powered by Gemini.",
    tags: ["Gemini", "Text chat", "Image generation"],
    demo: "chatbot.html",
  },
  {
    title: "YouTube Summarizer",
    kicker: "Video intelligence",
    description:
      "Paste a YouTube link and ask questions about it. Gemini watches the video and answers — no transcript scraping needed.",
    tags: ["Gemini", "YouTube", "Q&A"],
    demo: "youtube.html",
  },
];

const grid = document.querySelector("#project-grid");
const template = document.querySelector("#project-card-template");

projects.forEach((project) => {
  const card = template.content.cloneNode(true);
  const kicker = card.querySelector(".card-kicker");
  const title = card.querySelector("h3");
  const description = card.querySelector(".card-description");
  const tagList = card.querySelector(".tag-list");
  const demoLink = card.querySelector(".card-demo");

  kicker.textContent = project.kicker;
  title.textContent = project.title;
  description.textContent = project.description;
  demoLink.href = project.demo;
  demoLink.textContent = "Launch demo →";

  project.tags.forEach((tag) => {
    const item = document.createElement("li");
    item.textContent = tag;
    tagList.appendChild(item);
  });

  grid.appendChild(card);
});

const yearEl = document.querySelector("#year");
if (yearEl) yearEl.textContent = new Date().getFullYear();