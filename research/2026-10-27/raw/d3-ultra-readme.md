---
pipeline_tag: zero-shot-classification
license: apache-2.0
library_name: transformers
tags:
- zero-shot-classification
- decision-model
- classification
- system-one
- multimodal
- vision
- video
- object-detection
- safetensors
---

![d3-ultra: Towards Open Multimodal Foundation Decision Models](assets/banner@2x.png)

# d3-ultra

**d3-ultra** is the newest 27B multimodal decision model of Decision 3.0, the decision models of [vLLM Semantic Router](https://github.com/vllm-project/semantic-router). Give it an input (text or JSON, optionally with images and videos) and the questions you need answered: pick one of several options, say yes or no, rate on a scale, or point to where something is in an image or a video. It reads the input once per request, answers every question on it, and returns a probability for every answer, without generating text.

**Try it free** at [decision.vllm-sr.ai](https://decision.vllm-sr.ai): the free API serves d3-ultra · Official Space: [vllm-sr/d3-ultra](https://huggingface.co/spaces/vllm-sr/d3-ultra)

| | |
| --- | --- |
| **Parameters** | 26.09B, including the 0.46B vision encoder |
| **Inputs** | Text or JSON, plus images and videos (several per request); images can also sit inside questions and options |
| **Decision types** | Choice · Yes / No · Score · Box (images and video moments) · Track (whole videos) |
| **Context** | 131,072 tokens per question |
| **License** | Apache-2.0 |

## Highlights

- **State once:** d3-ultra reads the input once and answers every question from it: asking 1 to 255 questions about the same photo takes 304 ms to 2.16 s (median; one AMD Instinct MI325X, AutoModel; [Serving performance](#serving-performance)).
- **Long context:** up to 131,072 tokens per question, served on a single AMD Instinct MI325X.
- **Box:** ask where something is, in plain words, and get one box with a confidence that says whether it is there, answered from the same read of the picture as every other question.
- **Video:** a box at any moment of a video, or a track over the whole clip with the probability that the target is visible; the free API answers both.


## Examples

Real answers of this release, drawn as returned. The [Quickstart](#quickstart) below runs these same requests.

![One painting, ten questions: ten questions about Seurat's A Sunday on La Grande Jatte in one request, every option with its probability](assets/examples/grande-jatte.png)

Ten questions about Seurat's *A Sunday on La Grande Jatte*, answered from one read of the painting.

![A selfie from Mars: eight questions about a self-portrait of the Curiosity rover in one request, every option with its probability](assets/examples/curiosity.png)

Eight questions about the Curiosity rover's self-portrait on Mars, in one request.

![Pass, shoot or dribble? Five questions about a frame of a top-down football game in one request, every option with its probability](assets/examples/five-a-side.png)

A frame of a top-down football game: the next move, the free player, the pressure, the score and the keeper, in one request.

![Photo to etching: which 18th-century etching shows the same building as the photo? The question holds the photo and each option is an etching](assets/examples/pantheon.png)

The question holds a photo of the Pantheon, and each option is an 18th-century etching by Piranesi.

![Which planet is it? The question holds a picture of Saturn and each option is a picture of a planet](assets/examples/saturn.png)

The question holds a picture of Saturn, and each option is a picture of a planet.

![Red alert on the starship: five questions about one text message in one request, every option with its probability](assets/examples/starship.png)

Text only: five questions about one message to a starship's computer, in one call.

### Box: find it in the picture

Ask where something is and d3-ultra answers with one box and a confidence, the probability that the box is on what you asked for. All the Box questions about a picture are answered from one read of it, alongside any choice, yes / no or score questions in the same request, and a confidence near zero says the thing is not there.

![A cat in the canoe: five Box questions about George Caleb Bingham's Fur Traders Descending the Missouri in one request, each answer one box with its confidence](assets/examples/fur-traders.png)

The cat, the paddler, the boy and the canoe in George Caleb Bingham's *Fur Traders Descending the Missouri*, and a dog that is not there.

![Caught in the rain: five Box questions about Hiroshige's Sudden Shower over Shin-Ōhashi Bridge and Atake in one request, each answer one box with its confidence](assets/examples/sudden-shower.png)

The bridge, the raft, the people under umbrellas and the title in Hiroshige's woodblock print *Sudden Shower over Shin-Ōhashi Bridge and Atake*, and no sun in the rain.

![Parked on the Moon: five Box questions about an Apollo 15 photo in one request, each answer one box with its confidence](assets/examples/lunar-rover.png)

A photo from the Moon: the rover, the astronaut, the dish antenna and the mountain, and a flag that is not in this picture.

## Quickstart

```bash
pip install "transformers==5.17.0" torch torchvision pillow opencv-python-headless safetensors accelerate
pip install flash-linear-attention  # optional: fast GPU kernels for the linear-attention layers
```

```python
from huggingface_hub import hf_hub_download
from transformers import AutoModel

model = AutoModel.from_pretrained("vllm-sr/d3-ultra", trust_remote_code=True)


def asset(name):
    return hf_hub_download("vllm-sr/d3-ultra", f"assets/{name}")


# Text: several questions, one call
print(model.system_one(
    state="Ticket: the invoice total is wrong after the plan change.",
    questions={
        "route": {"type": "choice", "instructions": "Which team handles this?",
                  "criteria": {"billing": "Billing", "tech": "Technical support"}},
        "urgent": {"type": "noul", "instructions": "Is the customer asking for an urgent fix?"},
    },
))

# One painting, ten questions: the picture is read once for all of them
print(model.system_one(
    state="A painting of people relaxing in a park by a river.",
    images=[asset("grande-jatte.jpg")],
    questions={
        "style": {"type": "choice", "instructions": "Which painting style is this?",
                  "criteria": {"pointillism": "Pointillism, dots of pure colour", "cubism": "Cubism",
                               "watercolour": "A watercolour wash", "pop": "Pop art"}},
        "era": {"type": "choice", "instructions": "When was it most likely painted?",
                "criteria": {"1500s": "The 1500s", "1700s": "The 1700s", "1880s": "The 1880s", "1960s": "The 1960s"}},
        "weather": {"type": "choice", "instructions": "What is the weather like?",
                    "criteria": {"sunny": "Sunny", "rainy": "Rainy", "snowy": "Snowy", "foggy": "Foggy"}},
        "tophat": {"type": "noul", "instructions": "Is anyone wearing a top hat?"},
        "child": {"type": "noul", "instructions": "Is there a child in the picture?"},
        "parasol": {"type": "noul", "instructions": "Is anyone holding a parasol?"},
        "sailboat": {"type": "noul", "instructions": "Is there a sailboat on the river?"},
        "music": {"type": "noul", "instructions": "Is anyone playing a musical instrument?"},
        "fishing": {"type": "noul", "instructions": "Is anyone fishing?"},
        "swimming": {"type": "noul", "instructions": "Is anyone swimming in the river?"},
    },
))

# A selfie from Mars: eight questions about one picture
print(model.system_one(
    state="A self-portrait of a rover on another planet.",
    images=[asset("curiosity.jpg")],
    questions={
        "planet": {"type": "choice", "instructions": "Which planet is this?",
                   "criteria": {"mars": "Mars", "moon": "The Moon", "venus": "Venus", "earth": "Earth"}},
        "sky": {"type": "choice", "instructions": "What colour is the sky?",
                "criteria": {"blue": "Blue", "tan": "Hazy tan", "black": "Black", "green": "Green"}},
        "who": {"type": "choice", "instructions": "Who took this picture?",
                "criteria": {"rover": "The rover itself, with a camera on its arm", "astronaut": "An astronaut",
                             "satellite": "A satellite in orbit", "drone": "A drone"}},
        "butte": {"type": "noul", "instructions": "Is there a tall layered rock behind the rover?"},
        "ripples": {"type": "noul", "instructions": "Are there sand ripples on the ground?"},
        "wheels": {"type": "noul", "instructions": "Can you see the rover's wheels?"},
        "water": {"type": "noul", "instructions": "Is there liquid water anywhere in the picture?"},
        "person": {"type": "noul", "instructions": "Is there a person in the picture?"},
    },
))

# Pass, shoot or dribble? A game frame and five questions
print(model.system_one(
    state="A frame of Five-a-side, a small top-down football game: Blue attacks to the right, and blue "
          "number 8 has the ball.",
    images=[asset("five-a-side.jpg")],
    questions={
        "next": {"type": "choice", "instructions": "What should blue number 8 do next?",
                 "criteria": {"pass11": "Pass to number 11, free on the right wing",
                              "shoot": "Shoot at goal from where he is",
                              "dribble": "Dribble straight on into the red players",
                              "back": "Pass back to the goalkeeper"}},
        "free": {"type": "noul", "instructions": "Is blue number 11 free, with no red player close to him?"},
        "pressure": {"type": "noul", "instructions": "Is a red player right next to blue number 8?"},
        "score": {"type": "choice", "instructions": "What is the score?",
                  "criteria": {"0-0": "0 - 0", "1-0": "1 - 0", "0-1": "0 - 1", "2-1": "2 - 1"}},
        "keeper": {"type": "noul", "instructions": "Is the red goalkeeper inside his penalty area?"},
    },
))

# Photo to etching: the question holds a photo, each option is an etching
print(model.system_one(
    state="An architecture quiz.",
    questions={"pick": {"type": "choice", "instructions": "Which 18th-century etching shows the same building as <building>?",
                        "criteria": {"a": "<a>", "b": "<b>", "c": "<c>", "d": "<d>"}}},
    assets={"<building>": asset("pantheon.jpg"), "<a>": asset("pantheon-a.jpg"), "<b>": asset("pantheon-b.jpg"),
            "<c>": asset("pantheon-c.jpg"), "<d>": asset("pantheon-d.jpg")},
))

# Which planet is it? The question holds a picture, each option is one
print(model.system_one(
    state="A planet quiz.",
    questions={"pick": {"type": "choice", "instructions": "Which picture shows the same planet as <planet>?",
                        "criteria": {"a": "<a>", "b": "<b>", "c": "<c>"}}},
    assets={"<planet>": asset("saturn.jpg"), "<a>": asset("saturn-a.jpg"),
            "<b>": asset("saturn-b.jpg"), "<c>": asset("saturn-c.jpg")},
))

# Red alert on the starship: a text-only routing decision, five questions in one call
print(model.system_one(
    state="Message to the ship's computer: Captain, the reactor is overheating, the shields are down, and two crew "
          "members in the engine room are hurt.",
    questions={
        "team": {"type": "choice", "instructions": "Which team should fix the reactor?",
                 "criteria": {"engineering": "Engineering", "medical": "Medical", "navigation": "Navigation",
                              "galley": "The galley"}},
        "urgency": {"type": "choice", "instructions": "How urgent is this?",
                    "criteria": {"low": "It can wait until morning", "medium": "Handle it today",
                                 "high": "Right now: all hands"}},
        "injured": {"type": "noul", "instructions": "Is anyone hurt?"},
        "medics": {"type": "noul", "instructions": "Should a medical team go to the engine room?"},
        "danger": {"type": "noul", "instructions": "Is the ship in danger?"},
    },
))

# Box: who is in the canoe? Five questions, one read of the painting
print(model.system_one(
    state="A painting of a canoe on a river.",
    images=[asset("fur-traders.jpg")],
    questions={
        "cat": {"type": "box", "instructions": "The black cat"},
        "paddler": {"type": "box", "instructions": "The man with the paddle"},
        "boy": {"type": "box", "instructions": "The boy leaning on the cargo"},
        "canoe": {"type": "box", "instructions": "The canoe"},
        "dog": {"type": "box", "instructions": "A dog"},
    },
))

# Box: a woodblock print in the rain, five questions in one request
print(model.system_one(
    state="A Japanese woodblock print of a bridge in the rain.",
    images=[asset("sudden-shower.jpg")],
    questions={
        "bridge": {"type": "box", "instructions": "The bridge"},
        "raft": {"type": "box", "instructions": "The raft on the river"},
        "umbrellas": {"type": "box", "instructions": "The people under umbrellas"},
        "title": {"type": "box", "instructions": "The title cartouche"},
        "sun": {"type": "box", "instructions": "The sun"},
    },
))

# Box: a photo from the Moon, five questions in one request
print(model.system_one(
    state="A photo taken on the Moon.",
    images=[asset("lunar-rover.jpg")],
    questions={
        "rover": {"type": "box", "instructions": "The lunar rover"},
        "astronaut": {"type": "box", "instructions": "The astronaut"},
        "antenna": {"type": "box", "instructions": "The dish antenna"},
        "mountain": {"type": "box", "instructions": "The mountain behind the rover"},
        "flag": {"type": "box", "instructions": "A flag"},
    },
))

# Or as a pipeline:
# transformers.pipeline("decision", model="vllm-sr/d3-ultra", trust_remote_code=True)(state=..., questions=..., images=..., videos=...)
```

Images can be local paths, http(s) URLs, PIL images or base64-encoded images; videos, local paths, http(s) URLs, base64-encoded videos or frame arrays. A Box answer is `{"type": "box", "box": {"top", "bottom", "left", "right"}, "confidence"}`, the edges as fractions of the image's height and width. There is always exactly one box, so read its confidence: the figures above draw a box solid from 0.50 and dashed from 0.10, and below 0.10 they list the thing as not found. A question over 131,072 tokens is answered with the error `max_length_exceeded`, and nothing is truncated.

## Serving performance

**Ask once, answer many.** d3-ultra reads the input once per request and answers every question on it, so more questions about the same input add little time. On one AMD Instinct MI325X with AutoModel, one question about a photo takes 304 ms and 255 questions take 2.16 s: each question after the first adds about 7.3 ms.

![Ask once, answer many: median time to answer 1, 4, 16, 64, 128 and 255 questions about one photo, d3-ultra on one AMD Instinct MI325X](assets/ask-once.png)

| Questions about one photo | d3-ultra, p50 / p90 |
| ---: | ---: |
| 1 | 304 / 304 ms |
| 4 | 332 / 332 ms |
| 16 | 429 / 429 ms |
| 64 | 873 / 874 ms |
| 128 | 1.30 / 1.30 s |
| 255 | 2.16 / 2.20 s |

## Evaluation

We compare d3-ultra with GLiDE V1.2, d1-ultra, d3-turbo, d3 and Jev 1.14 on two tests of our own: **SR-Text** for text decisions and **SR-Box** for Box answers.

![SR-Text and SR-Box per model. SR-Text score, the mean of the six parts: d3-ultra 87.2, GLiDE V1.2 85.1, d1-ultra 83.4, d3-turbo 83.1, d3 81.4, Jev 1.14 73.1. SR-Box, 10 domains: d3-ultra 58.2%, d3-turbo 60.0%, Jev 1.14 45.7%; GLiDE V1.2, d1-ultra and d3 do not support Box questions](assets/eval-overall.png)

| Model | SR-Text, mean of six parts | SR-Box, 10 domains | SR-Box, 9 domains |
| --- | ---: | ---: | ---: |
| **d3-ultra** | **87.2** | **58.2%** | **54.6%** |
| GLiDE V1.2 | 85.1 | not supported | not supported |
| d1-ultra | 83.4 | not supported | not supported |
| d3-turbo | 83.1 | 60.0% | 59.6% |
| d3 | 81.4 | not supported | not supported |
| Jev 1.14 | 73.1 | 45.7% | 46.9% |

**SR-Text** has 10,411 text requests with 73,495 questions. Most re-create the skills of 34 public benchmarks with new items, in five capability areas: knowledge and reasoning, language understanding, retrieval and classification, tools and automation, and arts and human taste. The other 4,896 questions come from 43 task families written for this test, such as evaluating SQL queries and spreadsheet formulas, matching invoices, billing by time of use and routing support tickets. The SR-Text score is the mean of the six parts, each in chance-corrected accuracy ×100: 0 is chance and 100 is every answer right. d3-ultra scores highest in every area except arts and human taste, where d1-ultra is slightly ahead (90.8 vs 89.6), within the test's margin of error.

![SR-Text by capability area, chance-corrected, for d3-ultra, GLiDE V1.2, d1-ultra, d3-turbo, d3 and Jev 1.14. Knowledge and reasoning: d3-ultra 80.2, GLiDE V1.2 79.0, d1-ultra 72.5, d3-turbo 75.1, d3 70.0, Jev 1.14 63.9; Language understanding: d3-ultra 90.3, GLiDE V1.2 87.7, d1-ultra 86.9, d3-turbo 86.9, d3 83.7, Jev 1.14 77.2; Retrieval and classification: d3-ultra 97.7, GLiDE V1.2 94.2, d1-ultra 96.6, d3-turbo 94.6, d3 94.0, Jev 1.14 95.8; Tools and automation: d3-ultra 97.8, GLiDE V1.2 97.4, d1-ultra 93.8, d3-turbo 96.7, d3 94.0, Jev 1.14 89.0; Arts and human taste: d3-ultra 89.6, GLiDE V1.2 89.0, d1-ultra 90.8, d3-turbo 86.7, d3 85.7, Jev 1.14 67.8; New domains: d3-ultra 67.4, GLiDE V1.2 63.3, d1-ultra 60.0, d3-turbo 58.5, d3 60.9, Jev 1.14 44.9](assets/eval-sr-text.png)

**SR-Box** has 5,678 Box questions on 4,984 images. 4,778 ask for something in the picture, in ten domains: photos (one object of its kind, one among several, small objects, groups), art, referring expressions, receipts, documents, UI screens and charts; the other 900 ask for something that is not there and are not part of the score. A box counts as correct when it overlaps the reference box with an IoU of at least 0.5. The score is the share of correct boxes in each domain, averaged over the ten domains; the 9-domain score leaves out charts. d3-turbo remains our strongest Box model on natural images; d3-ultra leads on charts. GLiDE V1.2, d1-ultra and d3 do not support Box questions, so they have no SR-Box score (not a zero).

![SR-Box by domain for d3-ultra, d3-turbo and Jev 1.14: the share of correct boxes in each of ten domains and the 10- and 9-domain means](assets/eval-sr-box.png)

Both question sets are private: neither is published.

**Method.** These are our own tests, not an official leaderboard, with the same items and scorer for every model. d3-ultra (this release), d3-turbo and d3 (release v3.1.0) run from their model files; GLiDE V1.2 runs from its public model files (fastino/GLiDE, revision a7e45896) on our own hardware, measured on 23 October 2026, with no invalid answers; Jev 1.14 runs through TypeSafe's API as model jev-1.14.0, a pinned version, and d1-ultra through Liquid AI's API, which offers no version pin, both on 26 October 2026. Jev 1.14's invalid answers (to 86 of the 10,411 SR-Text requests) and the images its API refused for their size (816 of the 5,678 SR-Box questions) count as wrong.

## License

Apache-2.0 ([LICENSE](LICENSE)).

## Citation

```bibtex
@misc{d3ultra_2026,
  title        = {{d3-ultra}: A Multimodal Decision Model},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/d3-ultra}}
}
```
