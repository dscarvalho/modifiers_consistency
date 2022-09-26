import json
from dynaconf import settings
from tqdm import tqdm
from nltk import CFG
from nltk.parse.generate import generate
from data_access.adjectives_nouns import Adjectives, Nouns


def main():
    adjectives = Adjectives()
    nouns = Nouns()

    grammar = CFG.fromstring(f"""
    S -> ADJS NOUN
    ADJS -> ADJS ADJ | ADJ |
    ADJ -> {" | ".join([f"'{adj[0]}'" for adj in adjectives])}
    NOUN -> {" | ".join([f"'{noun[0]}'" for noun in nouns])} |
    """)

    phrases = set()
    for derivation in tqdm(generate(grammar, depth=5), desc="Generating phrases"):
        phrase = list()
        words = set()
        for word in derivation:
            if (word not in words):
                phrase.append(word)
                words.add(word)

        print(phrase)
        idxs = tuple([adjectives[w] if w in adjectives else nouns[w] for w in phrase])
        phrases.add(idxs)

    with open(settings["phrase_file_path"], "w") as outfile:
        json.dump(list(phrases), outfile)





if __name__ == '__main__':
    main()

