import os
import json
import tqdm
import random
import requests
import argparse
from multiprocessing import Pool

random.seed(42)
os.makedirs("paradigm-generations", exist_ok=True)

N_GENERATIONS = 7000
SKIP = 0
NUM_PROCESSES = 256

NUM_LEMMAS=5

GPT_OSS_ENDPOINT="http://localhost:8000/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
}

PROMPT = """You are an expert linguist tasked with generating high-quality pre-training data for a small language model that specializes in formal linguistic competence (i.e., the knowledge of rules and statistical regularities of language).

You will be given:
- A linguistic paradigm along with its syntactic template.
- A target genre and a sub-genre.
- A set of lemmas that must appear in the text.

Your task is to generate a natural, fluent text that must include a sentence that demonstrates the given paradigm, and reads like a genuine piece of writing in the specified genre/sub-genre.

PARADIGM: {paradigm}
TEMPLATE: {template}

TARGET GENRE: {genre}
TARGET SUB-GENRE: {subgenre}

REQUIRED LEMMAS:
{formatted_lemmas}

INSRTUCTIONS:
- Do **not** mention grammar, linguistics, rules, or examples.
- Do **not** explain or comment on any construction.
- Avoid didactic tone, meta-language, or contrived sentence patterns.
- Preserve coherence, plausibility, and stylistic consistency appropriate to the genre.

Now go, generate the text.

OUTPUT:
"""

def get_response(prompt):
    response = requests.post(
        url=GPT_OSS_ENDPOINT,
        headers=HEADERS, 
        data=json.dumps({
            "model": "openai/gpt-oss-120b",
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.95,
        })
    )
    reasoning_chain=response.json()["choices"][0]["message"]["reasoning_content"]
    return reasoning_chain, response.json()["choices"][0]["message"]["content"]

def generate_text(ind_specification):
    ind, specification = ind_specification
    prompt = PROMPT.format(
        paradigm=specification["paradigm"],
        template=specification["template"],
        genre=specification["genre"],
        subgenre=specification["subgenre"],
        formatted_lemmas=specification["formatted_lemmas"],
    )
    try:
        reasoning_chain, response = get_response(prompt)
        return ind, (specification, reasoning_chain, response)
    except Exception as e:
        print(f"Error generating text for {ind}: {e}")
        return ind, (specification, None, None)
    
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--paradigm", type=str, required=True)
    args=parser.parse_args()
    return args

def main():
    args=parse_args()
    paradigm=args.paradigm

    with open("blimp_paradigm_templates.json", "r") as f:
        blimp_paradigm_templates=json.load(f)
    paradigm_to_template={paradigm["paradigm"]: paradigm["template"] for paradigm in blimp_paradigm_templates}
    template=paradigm_to_template[paradigm]

    with open("genres.json", "r") as f:
        all_genres = json.load(f)
    with open("lemmas.json", "r") as f:
        all_lemmas = json.load(f)[:10000]

    specifications = []
    for i in range(N_GENERATIONS):
        # select a random genre
        genre = random.choice(all_genres)
        # select a random subgenre
        subgenre = random.choice(genre["subgenres"])
        genre=genre["genre"]
        # select 3 random lemmas
        lemmas = random.sample(all_lemmas, NUM_LEMMAS)

        formatted_lemmas = "\n".join([f"{i+1}. {lemma}" for i, lemma in enumerate(lemmas)])

        specifications.append(
            (
                i, 
                {
                    "paradigm": paradigm,
                    "template": template,
                    "genre": genre,
                    "subgenre": subgenre,
                    "lemmas": lemmas,
                    "formatted_lemmas": formatted_lemmas,
                }
            )
        )
    
    specifications = specifications[SKIP:]
    

    with Pool(processes=NUM_PROCESSES) as p:
        generations = list(tqdm.tqdm(p.imap(generate_text, specifications), total=len(specifications)))
    
    # sort generations by index
    generations = sorted(generations, key=lambda x: x[0])

    final_data=[]
    for ind, (specification, reasoning_chain, text) in generations:
        # remove formatted_skills and formatted_lemmas from specification
        specification = {k: v for k, v in specification.items() if k not in ["formatted_lemmas"]}
        final_data.append({
            "id": ind,
            "specification": specification,
            "reasoning_chain": reasoning_chain,
            "text": text
        })

    with open(f"paradigm-generations/{paradigm}_{SKIP}_{N_GENERATIONS}.json", "w") as f:
        json.dump(final_data, f, indent=4)

if __name__=="__main__":
    main()